#!/usr/bin/env python3
"""
OpenAI-compatible API server for Cooperative LoRA model.

Supports multi-GPU: loads one model per GPU and round-robins requests.

Usage:
  python evaluation/serve_cooperative.py \
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
      --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_thought_v1/final \
      --num_gpus 4 --port 8000
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid
from io import BytesIO

import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from typing import Any, Dict, List

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper


# ── Request/Response models ────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = "cooperative"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Dict[str, str]
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


# ── Multi-GPU model pool ──────────────────────────────────────────

class ModelPool:
    """Thread-safe pool of models, one per GPU. Each GPU has its own lock."""

    def __init__(self, base_model_path, coop_checkpoint_path, num_gpus):
        self.models = []
        self.processors = []
        self.devices = []
        self.locks = []

        # Read config once
        config_path = os.path.join(coop_checkpoint_path, "cooperative_config.json")
        with open(config_path) as f:
            coop_config = json.load(f)
        lora_v_state = torch.load(
            os.path.join(coop_checkpoint_path, "lora_v.pt"),
            map_location="cpu", weights_only=True)
        first_A = [v for k, v in lora_v_state.items() if "lora_A_v" in k][0]
        r = first_A.shape[0]
        alpha = r * 2

        for gpu_id in range(num_gpus):
            device = f"cuda:{gpu_id}"
            print(f"Loading model on {device}...")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device,
            )
            model = CooperativeVLMWrapper(
                base_model=base_model,
                lora_r=r,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=coop_config["target_modules"],
                bind_weight=0.0,
            )
            model.load_cooperative_checkpoint(coop_checkpoint_path)
            model.eval()

            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

            self.models.append(model)
            self.processors.append(processor)
            self.devices.append(base_model.device)
            self.locks.append(threading.Lock())

        self._counter = 0
        self._counter_lock = threading.Lock()
        print(f"Model pool ready: {num_gpus} GPUs")

    def acquire(self):
        """Get the next available (model, processor, device) with its lock held."""
        with self._counter_lock:
            idx = self._counter % len(self.models)
            self._counter += 1

        self.locks[idx].acquire()
        return idx, self.models[idx], self.processors[idx], self.devices[idx]

    def release(self, idx):
        self.locks[idx].release()


# ── Inference ──────────────────────────────────────────────────────

def process_messages(messages, processor, model, device, max_tokens=512, temperature=0.0):
    """Convert OpenAI-format messages to model input and run inference."""
    qwen_messages = []
    images = []

    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        image_url = part["image_url"]
                        url = image_url if isinstance(image_url, str) else image_url.get("url", "")

                        if url.startswith("data:"):
                            header, b64data = url.split(",", 1)
                            img_bytes = base64.b64decode(b64data)
                            image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        elif url.startswith("file://"):
                            image = Image.open(url[7:]).convert("RGB")
                        elif os.path.exists(url):
                            image = Image.open(url).convert("RGB")
                        else:
                            continue

                        images.append(image)
                        parts.append({"type": "image", "image": image})
                elif isinstance(part, str):
                    parts.append({"type": "text", "text": part})
            qwen_messages.append({"role": role, "content": parts})

    text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)

    if images:
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI()
pool: ModelPool = None
executor: ThreadPoolExecutor = None


def _sync_inference(messages, max_tokens, temperature, model_name):
    """Run inference synchronously in a thread."""
    idx, model, processor, device = pool.acquire()
    try:
        response_text = process_messages(
            messages, processor, model, device,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    finally:
        pool.release(idx)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model_name,
        choices=[ChatCompletionChoice(
            message={"role": "assistant", "content": response_text},
        )],
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        _sync_inference,
        request.messages, request.max_tokens, request.temperature, request.model,
    )
    return result


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "cooperative", "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    global pool, executor

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    pool = ModelPool(args.base_model, args.coop_checkpoint, args.num_gpus)
    executor = ThreadPoolExecutor(max_workers=args.num_gpus)

    print(f"Server ready on {args.host}:{args.port} ({args.num_gpus} GPUs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
