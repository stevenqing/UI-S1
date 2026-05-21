#!/usr/bin/env python3
"""Serve V16 dual-prompt cooperative LoRA model via OpenAI-compatible API.

V16 inference flow (per request):
  1. Build grounder prompt from the same image/instruction/history
  2. Grounder forward (Expert 1 only) → cache per-layer hidden states
  3. Actor generate with cached grounder features → output action

Multi-GPU: loads one model replica per GPU for parallel inference.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

app = FastAPI()
WORKERS = []
PROCESSOR = None
MODEL_NAME = "v16"
EXECUTOR = ThreadPoolExecutor(max_workers=8)
WORKER_QUEUE = None

# ── Grounder prompt ──────────────────────────────────────────────

GROUNDER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen and user instruction, you need to output the position of the element you will operate.

The instruction is:
{instruction}

The history of actions are:
{history}

Output the coordinate of the element you will operate within <coordinate></coordinate> tag:
<coordinate> [x, y] </coordinate>"""


# ── Request / Response models ────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str = "v16"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


# ── Load models ──────────────────────────────────────────────────

def load_models(args):
    global WORKERS, PROCESSOR, MODEL_NAME
    MODEL_NAME = args.served_model_name or "v16"

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

        # Use DualPromptCooperativeWrapper from V16 train script
        from v16_task_vector_merge.train_dual_prompt_rl import DualPromptCooperativeWrapper
        model = DualPromptCooperativeWrapper(
            base_model,
            lora_r=coop_config.get("lora_r", 128),
            lora_alpha=coop_config.get("lora_alpha", 256),
            target_modules=coop_config.get("target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]),
            num_comm_rounds=coop_config.get("num_comm_rounds", 2),
            no_routing=args.no_routing,
        )

        model.load_cooperative(args.coop_checkpoint)
        model.eval()
        WORKERS.append((model, gpu_id))
        print(f"  GPU {gpu_id} ready")

    print(f"All {num_gpus} replicas loaded.")


# ── Extract instruction/history from actor messages ──────────────

def extract_instruction_and_history(messages):
    """Parse instruction and action history from the user message text."""
    text = ""
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                text = part["text"]

    instruction = ""
    history = "None"

    # Extract instruction
    m = re.search(r"The instruction is:\n(.*?)(?:\n\nThe history)", text, re.DOTALL)
    if m:
        instruction = m.group(1).strip()

    # Extract history
    m = re.search(r"The history of actions are:\n(.*?)(?:\n\nThe actions supported)", text, re.DOTALL)
    if m:
        history = m.group(1).strip()

    return instruction, history


# ── Process messages ─────────────────────────────────────────────

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
            processed.append({"role": msg.role, "content": parts})
        else:
            processed.append({"role": msg.role, "content": [{"type": "text", "text": str(msg.content)}]})
    return processed


# ── Inference ────────────────────────────────────────────────────

SKIP_GROUNDER = False  # Set via --no_grounder_cache flag


def _do_inference(worker_idx, messages, max_tokens, temperature, top_p):
    """V16 inference: grounder pass → cache → actor generate."""
    model, gpu_id = WORKERS[worker_idx]
    device = f"cuda:{gpu_id}"

    # Extract image from messages
    images = []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                img = part["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images.append(img)

    image = images[0] if images else None

    # Extract instruction and history for grounder prompt
    instruction, history = extract_instruction_and_history(messages)

    # Step 1: Grounder pass (cache hidden states)
    # When SKIP_GROUNDER=True, skip caching — Expert 1 uses actor input directly
    # (matches training behavior for non-coordinate steps)
    if not SKIP_GROUNDER and image is not None and instruction:
        grounder_text_content = GROUNDER_PROMPT_TEMPLATE.format(
            instruction=instruction, history=history
        )
        grounder_msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": grounder_text_content},
        ]}]
        grounder_text = PROCESSOR.apply_chat_template(
            grounder_msgs, tokenize=False, add_generation_prompt=True
        )
        grounder_inputs = PROCESSOR(
            text=[grounder_text], images=[image],
            return_tensors="pt", padding=False
        ).to(device)

        with torch.no_grad():
            model.cache_grounder_hidden_states(grounder_inputs)
    else:
        model.clear_grounder_cache()

    # Step 2: Actor generate
    actor_text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if images:
        inputs = PROCESSOR(text=[actor_text], images=images,
                          return_tensors="pt", padding=True).to(device)
    else:
        inputs = PROCESSOR(text=[actor_text], return_tensors="pt",
                          padding=True).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            do_sample=temperature > 0,
        )

    # Clear grounder cache
    model.clear_grounder_cache()

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    text_out = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text_out


# ── API endpoints ────────────────────────────────────────────────

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
        text_out = await loop.run_in_executor(
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
            message={"role": "assistant", "content": text_out},
        )],
    )


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--coop_checkpoint", type=str, required=True)
    parser.add_argument("--served-model-name", type=str, default="v16")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--no_routing", action="store_true", default=False)
    parser.add_argument("--no_grounder_cache", action="store_true", default=False,
                        help="Skip grounder caching — Expert 1 uses actor input directly")
    args = parser.parse_args()

    global SKIP_GROUNDER
    SKIP_GROUNDER = args.no_grounder_cache
    if SKIP_GROUNDER:
        print("*** Grounder caching DISABLED — Expert 1 uses actor input directly ***")

    load_models(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
