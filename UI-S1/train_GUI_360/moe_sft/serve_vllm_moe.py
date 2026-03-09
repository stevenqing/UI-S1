"""
vLLM Multi-LoRA MoE Server with per-step dynamic routing.

Uses a standalone SigLIP router (~100M params) to select expert LoRA
for each request, combined with vLLM's native multi-LoRA support
for high-throughput inference.

Architecture:
    Client Request (screenshot + prompt)
           │
           ▼
    ┌─────────────────────┐
    │  StandaloneRouter   │  SigLIP-based, ~100M params, <10ms
    │  predict(screenshot)│  → expert_idx
    └─────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  vLLM Engine        │  Base model + multi-LoRA
    │  generate(prompt,   │  → response
    │    lora=expert_idx) │
    └─────────────────────┘
           │
           ▼
    Response (action text)

Usage:
    python serve_vllm_moe.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
        --standalone_router standalone_router_checkpoint/best \
        --port 19810

The existing serve_moe.py (HF-based) is preserved for non-vLLM use cases.
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from io import BytesIO
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================
# Pydantic models for OpenAI API
# ============================================
class ChatMessage(PydanticModel):
    role: str
    content: object  # str or list of content parts

class ChatCompletionRequest(PydanticModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


# ============================================
# Global references
# ============================================
standalone_router = None
llm = None
expert_lora_requests = None
processor = None


def extract_screenshot_from_messages(messages: List[ChatMessage]):
    """Extract the first screenshot from messages as PIL Image."""
    from PIL import Image

    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image"):
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        return Image.open(BytesIO(img_bytes)).convert("RGB")
    return None


def build_prompt_from_messages(messages: List[ChatMessage]) -> str:
    """Build a text prompt from messages for vLLM."""
    formatted = []
    for msg in messages:
        if isinstance(msg.content, str):
            formatted.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        elif isinstance(msg.content, list):
            content_parts = []
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        content_parts.append({"type": "image", "image": "placeholder"})
                elif isinstance(part, str):
                    content_parts.append({"type": "text", "text": part})
            formatted.append({"role": msg.role, "content": content_parts})

    text = processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
    return text


def load_models(
    base_model_path: str,
    moe_checkpoint_path: str,
    standalone_router_path: str,
    max_loras: int = 8,
    gpu_memory_utilization: float = 0.85,
):
    """Load standalone router and vLLM engine with multi-LoRA."""
    global standalone_router, llm, expert_lora_requests, processor

    from transformers import AutoProcessor
    from verl.models.moe.standalone_router import StandaloneRouter

    # 1. Load standalone router
    logger.info(f"Loading standalone router from {standalone_router_path}...")
    standalone_router = StandaloneRouter.load(standalone_router_path, device="cuda")
    standalone_router.eval()
    logger.info("Standalone router loaded")

    # 2. Load processor for chat template
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # 3. Determine number of experts from MoE config
    config_path = os.path.join(moe_checkpoint_path, "moe_config.json")
    with open(config_path) as f:
        moe_config = json.load(f)
    num_experts = moe_config.get("num_experts", 6)
    lora_r = moe_config.get("expert_lora_r", 16)

    # 4. Build expert LoRA paths
    experts_dir = os.path.join(moe_checkpoint_path, "experts")
    expert_paths = []
    for i in range(num_experts):
        expert_dir = os.path.join(experts_dir, f"expert_{i}")
        if os.path.exists(expert_dir):
            expert_paths.append(expert_dir)
        else:
            logger.warning(f"Expert {i} not found at {expert_dir}")

    # 5. Start vLLM engine with multi-LoRA support
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        logger.info(f"Starting vLLM with base model: {base_model_path}")
        llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=lora_r,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        # Create LoRA requests for each expert
        expert_lora_requests = []
        for i, expert_path in enumerate(expert_paths):
            lora_req = LoRARequest(
                lora_name=f"expert_{i}",
                lora_int_id=i + 1,  # vLLM requires positive int IDs
                lora_path=expert_path,
            )
            expert_lora_requests.append(lora_req)

        logger.info(f"Loaded {len(expert_lora_requests)} expert LoRAs for vLLM")

    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        raise


def generate_response(
    messages: List[ChatMessage],
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """Generate response with per-step expert routing."""
    from vllm import SamplingParams

    # 1. Extract screenshot for routing
    screenshot = extract_screenshot_from_messages(messages)

    # 2. Route to expert via standalone router
    if screenshot is not None and standalone_router is not None:
        expert_idx, routing_weights = standalone_router.predict_with_distribution(screenshot)
        logger.info(f"Routed to expert_{expert_idx} (weights: {routing_weights.tolist()})")
    else:
        expert_idx = 0
        routing_weights = None
        logger.warning("No screenshot found, defaulting to expert_0")

    # 3. Build prompt
    prompt = build_prompt_from_messages(messages)

    # 4. Generate with vLLM using selected expert LoRA
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=1.0,
    )

    lora_request = expert_lora_requests[expert_idx] if expert_idx < len(expert_lora_requests) else None

    outputs = llm.generate(
        [prompt],
        sampling_params,
        lora_request=lora_request,
    )

    response_text = outputs[0].outputs[0].text

    return {
        "response": response_text,
        "expert_idx": expert_idx,
        "routing_weights": routing_weights.tolist() if routing_weights is not None else None,
    }


# ============================================
# FastAPI app
# ============================================
app = FastAPI(title="vLLM MoE Multi-LoRA Server")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "vllm"}


@app.get("/v1/models")
async def list_models():
    models = [{"id": "moe-vllm", "object": "model", "owned_by": "local"}]
    if expert_lora_requests:
        for req in expert_lora_requests:
            models.append({"id": req.lora_name, "object": "model", "owned_by": "local"})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        start = time.time()
        result = generate_response(
            request.messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.0,
        )
        elapsed = time.time() - start

        response_text = result["response"]
        logger.info(
            f"Generated in {elapsed:.2f}s, expert={result['expert_idx']}, "
            f"len={len(response_text)}"
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "moe-vllm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "routing_info": {
                "expert_idx": result["expert_idx"],
                "routing_weights": result["routing_weights"],
            },
        }
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="vLLM Multi-LoRA MoE Server")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--moe_checkpoint", type=str, required=True)
    parser.add_argument("--standalone_router", type=str, required=True,
                        help="Path to trained standalone router checkpoint")
    parser.add_argument("--port", type=int, default=19810)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max_loras", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    load_models(
        base_model_path=args.base_model,
        moe_checkpoint_path=args.moe_checkpoint,
        standalone_router_path=args.standalone_router,
        max_loras=args.max_loras,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
