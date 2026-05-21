#!/usr/bin/env python3
"""Serve V17 cooperative LoRA model via OpenAI-compatible API.

Extends V13 serve_cooperative_direct.py with V17 aux-decision awareness:
  - Loads V13, PhaseAware, or DualAux cooperative wrapper
  - Strips <aux>...</aux>, <aux_a>...</aux_a>, <aux_b>...</aux_b> from output by default
  - Reports aux content in metadata for analysis

Multi-GPU: loads one model replica per GPU for parallel inference.

Usage:
    python v17_self_evolve/serve_v17_direct.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v17_rl/epoch-0/cooperative \
        --port 8000

    # Keep aux in output (for analysis):
    python v17_self_evolve/serve_v17_direct.py \
        --base_model ... --coop_checkpoint ... --keep_aux
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel
from typing import List, Optional, Union
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

app = FastAPI()

# Global state
WORKERS = []
WORKER_QUEUE = None
EXECUTOR = ThreadPoolExecutor(max_workers=16)
PROCESSOR = None
MODEL_NAME = "v17-cooperative"
KEEP_AUX = False

_AUX_RE = re.compile(r"<aux>.*?</aux>\s*", re.DOTALL)
_AUX_A_RE = re.compile(r"<aux_a>.*?</aux_a>\s*", re.DOTALL)
_AUX_B_RE = re.compile(r"<aux_b>.*?</aux_b>\s*", re.DOTALL)


# -- Pydantic models ──────────────────────────────────────────────────

class ChatMessage(PydanticBaseModel):
    role: str
    content: Union[str, list]

class ChatCompletionRequest(PydanticBaseModel):
    model: str = "v17-cooperative"
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
    aux_content: Optional[str] = None
    aux_a_content: Optional[str] = None
    aux_b_content: Optional[str] = None


# -- Load models ──────────────────────────────────────────────────────

def load_models(args):
    global WORKERS, PROCESSOR, MODEL_NAME, KEEP_AUX
    MODEL_NAME = args.served_model_name or "v17-cooperative"
    KEEP_AUX = args.keep_aux

    num_gpus = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
    print(f"Loading {num_gpus} model replicas...")

    PROCESSOR = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    print(f"Cooperative config: {json.dumps(coop_config, indent=2)}")

    coop_type = coop_config.get("type", "iterative_cooperative_v13")

    for gpu_id in range(num_gpus):
        print(f"  Loading replica on GPU {gpu_id}...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map={"": gpu_id},
        )

        if coop_type == "dual_aux_cooperative_v17":
            from v17_self_evolve.dual_aux_cooperative_lora import (
                DualAuxCooperativeVLMWrapper,
            )
            model = DualAuxCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 128),
                lora_alpha=coop_config.get("lora_alpha", 256),
                target_modules=coop_config.get("target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
                num_comm_rounds=coop_config.get("num_comm_rounds", 2),
                aux_a_hard_route=coop_config.get("aux_a_hard_route", 0.9),
                aux_b_hard_route=coop_config.get("aux_b_hard_route", 0.1),
            )
        elif coop_type == "phase_aware_cooperative_v17":
            from v17_self_evolve.phase_aware_cooperative_lora import (
                PhaseAwareCooperativeVLMWrapper,
            )
            model = PhaseAwareCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 128),
                lora_alpha=coop_config.get("lora_alpha", 256),
                target_modules=coop_config.get("target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
                num_comm_rounds=coop_config.get("num_comm_rounds", 2),
            )
        else:
            from v13_gui_360.iterative_cooperative_wrapper import (
                IterativeCooperativeVLMWrapper,
            )
            model = IterativeCooperativeVLMWrapper(
                base_model,
                lora_r=coop_config.get("lora_r", 128),
                lora_alpha=coop_config.get("lora_alpha", 256),
                target_modules=coop_config.get("target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
                num_comm_rounds=coop_config.get("num_comm_rounds", 2),
            )

        model.load_cooperative(args.coop_checkpoint)

        if args.expert_mode is not None:
            model.set_inference_mode(args.expert_mode)
            print(f"  Expert mode: {args.expert_mode}")

        if args.no_comm:
            model.set_disable_comm(True)
            print(f"  Communication DISABLED")

        model.eval()
        WORKERS.append((model, gpu_id))
        print(f"  GPU {gpu_id} ready, params={model.count_trainable_params()}")

    print(f"All {num_gpus} replicas loaded. keep_aux={KEEP_AUX}")


# -- Process messages ─────────────────────────────────────────────────

def process_messages(messages):
    """Convert OpenAI-format messages to Qwen processor input."""
    processed = []
    for msg in messages:
        if isinstance(msg.content, str):
            processed.append({"role": msg.role,
                              "content": [{"type": "text", "text": msg.content}]})
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
            processed.append({"role": msg.role,
                              "content": [{"type": "text", "text": str(msg.content)}]})
    return processed


# -- Inference ────────────────────────────────────────────────────────

def _do_inference(worker_idx, messages, max_tokens, temperature, top_p):
    """Run inference on a specific GPU worker."""
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

    # Stop at </decision>
    tokenizer = PROCESSOR.tokenizer
    stop_ids = [tokenizer.eos_token_id]
    tc_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(tc_ids) == 1:
        stop_ids.append(tc_ids[0])

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=stop_ids,
            stop_strings=["</decision>"],
            tokenizer=tokenizer,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    text_out = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract aux content before stripping
    aux_content = None
    aux_a_content = None
    aux_b_content = None

    # Single <aux> tag (V17.0 format)
    aux_m = _AUX_RE.search(text_out)
    if aux_m:
        aux_content = aux_m.group(0).replace("<aux>", "").replace("</aux>", "").strip()

    # Dual <aux_a>/<aux_b> tags (V17.1 format)
    aux_a_m = _AUX_A_RE.search(text_out)
    if aux_a_m:
        aux_a_content = aux_a_m.group(0).replace("<aux_a>", "").replace("</aux_a>", "").strip()

    aux_b_m = _AUX_B_RE.search(text_out)
    if aux_b_m:
        aux_b_content = aux_b_m.group(0).replace("<aux_b>", "").replace("</aux_b>", "").strip()

    # Strip aux tags from output unless keep_aux is enabled
    if not KEEP_AUX:
        text_out = _AUX_RE.sub("", text_out)
        text_out = _AUX_A_RE.sub("", text_out)
        text_out = _AUX_B_RE.sub("", text_out).strip()

    return text_out, aux_content, aux_a_content, aux_b_content


# -- API endpoints ────────────────────────────────────────────────────

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

    worker_idx = await WORKER_QUEUE.get()
    try:
        loop = asyncio.get_event_loop()
        text_out, aux_content, aux_a_content, aux_b_content = await loop.run_in_executor(
            EXECUTOR, _do_inference, worker_idx, messages,
            request.max_tokens, request.temperature, request.top_p,
        )
    finally:
        WORKER_QUEUE.put_nowait(worker_idx)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=text_out),
        )],
        aux_content=aux_content,
        aux_a_content=aux_a_content,
        aux_b_content=aux_b_content,
    )


# -- Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPU replicas (0=auto-detect)")
    parser.add_argument("--keep_aux", action="store_true",
                        help="Keep <aux> tags in output (for analysis)")
    parser.add_argument("--expert_mode", type=str, default=None,
                        choices=["expert_1_only", "expert_2_only"])
    parser.add_argument("--no_comm", action="store_true",
                        help="Disable iterative communication")
    args = parser.parse_args()

    load_models(args)
    print(f"Starting server on port {args.port} with {len(WORKERS)} workers...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
