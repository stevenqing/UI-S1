"""
Lightweight OpenAI-compatible server for MoE model inference.

Loads the base model + MoE wrapper (router + expert LoRAs) and serves
an OpenAI-compatible /v1/chat/completions endpoint for evaluation.

Usage:
    python serve_moe.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v2/final \
        --port 19809
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

# Add project root to path
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
# Global model references
# ============================================
moe_wrapper = None
processor = None
device = None


def load_model(base_model_path: str, moe_checkpoint_path: str):
    """Load base model + MoE wrapper."""
    global moe_wrapper, processor, device

    from transformers import AutoProcessor, AutoModelForVision2Seq
    from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load MoE config
    config_path = os.path.join(moe_checkpoint_path, "moe_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        moe_config = MoEConfig(**config_dict)
        logger.info(f"Loaded MoE config: {moe_config.num_experts} experts, r={moe_config.expert_lora_r}")
    else:
        raise FileNotFoundError(f"MoE config not found at {config_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    logger.info("Processor loaded")

    # Load base model
    logger.info(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("Base model loaded")

    # Create MoE wrapper (wraps base model + creates MoELoRALinear modules)
    moe_wrapper = MoEVLMWrapper(
        base_model=base_model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )

    # Load MoE checkpoint (router + expert LoRAs)
    moe_wrapper.load_moe_checkpoint(moe_checkpoint_path)
    logger.info(f"MoE checkpoint loaded from {moe_checkpoint_path}")

    # Move to device with consistent dtype
    moe_wrapper = moe_wrapper.to(device=device, dtype=torch.bfloat16)
    moe_wrapper.eval()
    logger.info("Model ready for inference")


def extract_images_from_messages(messages: List[ChatMessage]) -> list:
    """Extract base64 images from message content."""
    from PIL import Image

    images = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image"):
                        # Extract base64 data
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        images.append(img)
    return images


def build_text_from_messages(messages: List[ChatMessage]) -> list:
    """Convert messages to the format expected by the processor."""
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
    return formatted


@torch.no_grad()
def generate_response(messages: List[ChatMessage], max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """Generate a response using the MoE model."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    # Build conversation format for processor
    formatted_messages = build_text_from_messages(messages)

    # Extract images
    images = extract_images_from_messages(messages)

    # Replace placeholder with actual image references
    img_idx = 0
    for msg in formatted_messages:
        for i, part in enumerate(msg["content"]):
            if part.get("type") == "image" and img_idx < len(images):
                msg["content"][i] = {"type": "image", "image": images[img_idx]}
                img_idx += 1

    # Apply chat template
    text = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)

    # Process vision info
    image_inputs, video_inputs = process_vision_info(formatted_messages)

    # Tokenize
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Extract instruction text for routing
    instruction_texts = []
    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                instruction_texts.append(msg.content)
            elif isinstance(msg.content, list):
                text_parts = [p["text"] for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
                instruction_texts.append(" ".join(text_parts))
    instruction_text = " ".join(instruction_texts)

    # Generate using MoE wrapper
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    generated_ids, router_output = moe_wrapper.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        instruction_texts=[instruction_text],
        **gen_kwargs,
    )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Log routing info
    weights = router_output.routing_weights[0].cpu().tolist()
    logger.debug(f"Routing weights: {[f'{w:.3f}' for w in weights]}")

    return response


# ============================================
# FastAPI app
# ============================================
app = FastAPI(title="MoE Model Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "moe-model", "object": "model", "owned_by": "local"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        start = time.time()
        response_text = generate_response(
            request.messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.0,
        )
        elapsed = time.time() - start
        logger.info(f"Generated response in {elapsed:.2f}s ({len(response_text)} chars)")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or "moe-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="MoE Model Server")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--moe_checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=19809)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model(args.base_model, args.moe_checkpoint)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
