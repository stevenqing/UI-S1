#!/usr/bin/env python3
"""Serve cooperative LoRA model directly via OpenAI-compatible API.

No merge/correction needed — loads base model + cooperative wrapper directly.
Works for both V12 (SoftCooperativeVLMWrapper) and V13 (IterativeCooperativeVLMWrapper).

Multi-GPU: loads one model replica per GPU for parallel inference.

Usage:
    # V12 (auto-detect GPUs)
    python v13_gui_360/serve_cooperative_direct.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v12_gui360_rl/epoch-0/cooperative \
        --version v12 --port 8000

    # V13, 4 GPUs
    python v13_gui_360/serve_cooperative_direct.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v13_gui360_rl/epoch-0/cooperative \
        --version v13 --port 8000 --num_gpus 4
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Union
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

app = FastAPI()

# ── Global state ─────────────────────────────────────────────────────
WORKERS = []       # [(model, gpu_id), ...]
WORKER_QUEUE = None  # asyncio.Queue — indexes into WORKERS
EXECUTOR = ThreadPoolExecutor(max_workers=16)
PROCESSOR = None
MODEL_NAME = "cooperative"


# ── Pydantic models for OpenAI-compatible API ────────────────────────

class ChatMessage(PydanticBaseModel):
    role: str
    content: Union[str, list]

class ChatCompletionRequest(PydanticBaseModel):
    model: str = "cooperative"
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0

class ChatCompletionChoice(PydanticBaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(PydanticBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


# ── Load models (one per GPU) ────────────────────────────────────────

def load_models(args):
    global WORKERS, PROCESSOR, MODEL_NAME
    MODEL_NAME = args.served_model_name or "cooperative"

    num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
    print(f"Loading {num_gpus} model replicas...")

    PROCESSOR = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    print(f"Cooperative config: {json.dumps(coop_config, indent=2)}")

    for gpu_id in range(num_gpus):
        print(f"  Loading replica on GPU {gpu_id}...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map={"": gpu_id},
        )

        if args.version == "v14":
            from v14_gui_360.cooperative_wrapper import IterativeCooperativeVLMWrapper
            model = IterativeCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 128),
                lora_alpha=coop_config.get("lora_alpha", 256),
                target_modules=coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                num_comm_rounds=coop_config.get("num_comm_rounds", 2),
            )
        elif args.version == "v13":
            from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper
            model = IterativeCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 128),
                lora_alpha=coop_config.get("lora_alpha", 256),
                target_modules=coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                num_comm_rounds=coop_config.get("num_comm_rounds", 2),
            )
        else:
            from v12_gui_360.soft_cooperative_wrapper import SoftCooperativeVLMWrapper
            model = SoftCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 256),
                lora_alpha=coop_config.get("lora_alpha", 512),
                target_modules=coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )

        model.load_cooperative(args.coop_checkpoint)

        # Expert-only ablation: force single expert output
        if args.expert_mode is not None:
            model.set_inference_mode(args.expert_mode)
            print(f"  Expert mode: {args.expert_mode}")

        # Disable communication (ablation)
        if args.no_comm:
            model.set_disable_comm(True)
            print(f"  Communication DISABLED")

        # Layer-selective evaluation: only enable LoRA for specified layers
        if args.active_layers is not None:
            active = [int(x) for x in args.active_layers.split(",")]
            model.enable_lora_layers(active)
            print(f"  Layer-selective mode: only layers {active} active")

        model.eval()
        WORKERS.append((model, gpu_id))
        print(f"  GPU {gpu_id} ready, params={model.count_trainable_params()}")

    print(f"All {num_gpus} replicas loaded.")


# ── Process messages ──────────────────────────────────────────────────

def process_messages(messages):
    """Convert OpenAI-format messages to Qwen processor input."""
    processed = []
    for msg in messages:
        if isinstance(msg.content, str):
            processed.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        elif isinstance(msg.content, list):
            parts = []
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            parts.append({"type": "image", "image": img})
                        else:
                            parts.append({"type": "image", "image": url})
                else:
                    parts.append({"type": "text", "text": str(item)})
            processed.append({"role": msg.role, "content": parts})
        else:
            processed.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})
    return processed


# ── Inference on a specific worker ────────────────────────────────────

def _do_inference(worker_idx, messages, max_tokens, temperature, top_p):
    """Run inference on a specific GPU worker (called from thread pool)."""
    model, gpu_id = WORKERS[worker_idx]
    device = f"cuda:{gpu_id}"

    text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    images = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                img = part["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images.append(img)

    if images:
        inputs = PROCESSOR(text=[text], images=images, return_tensors="pt",
                          padding=True).to(device)
    else:
        inputs = PROCESSOR(text=[text], return_tensors="pt",
                          padding=True).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            do_sample=temperature > 0,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    text_out = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text_out


# ── API endpoints ─────────────────────────────────────────────────────

@app.on_event("startup")
async def init_worker_queue():
    global WORKER_QUEUE
    WORKER_QUEUE = asyncio.Queue()
    for i in range(len(WORKERS)):
        WORKER_QUEUE.put_nowait(i)
    print(f"Worker queue initialized with {len(WORKERS)} workers")

@app.get("/health")
def health():
    return {"status": "ok", "workers": len(WORKERS)}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = process_messages(request.messages)

    # Acquire a free worker (blocks if all busy)
    worker_idx = await WORKER_QUEUE.get()
    try:
        loop = asyncio.get_event_loop()
        text_out = await loop.run_in_executor(
            EXECUTOR, _do_inference, worker_idx, messages,
            request.max_tokens, request.temperature, request.top_p,
        )
    finally:
        # Always return worker to pool
        WORKER_QUEUE.put_nowait(worker_idx)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=text_out),
        )],
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--version", choices=["v12", "v13", "v14"], required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPU replicas (0=auto-detect)")
    parser.add_argument("--active_layers", type=str, default=None,
                        help="Comma-separated layer indices to keep active (e.g. '10' or '0,1,2,3'). "
                             "All other layers use base model only. Default: all layers active.")
    parser.add_argument("--expert_mode", type=str, default=None,
                        choices=["expert_1_only", "expert_2_only"],
                        help="Force single expert output. Default: normal soft routing.")
    parser.add_argument("--no_comm", action="store_true",
                        help="Disable iterative communication between experts.")
    args = parser.parse_args()

    load_models(args)
    print(f"Starting server on port {args.port} with {len(WORKERS)} workers...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
