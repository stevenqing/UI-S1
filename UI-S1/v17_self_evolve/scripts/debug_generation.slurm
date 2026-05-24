#!/bin/bash
#SBATCH --job-name=v17_debug
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/debug_gen_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/debug_gen_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -e
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_DIR"

python3 << 'PYEOF'
import torch, json, sys, os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

model_path = "checkpoints/Qwen2.5-VL-7B-Instruct"
sft_ckpt = "checkpoints/v17_dual_aux_sft_warmup/epoch-0/cooperative"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

from v17_self_evolve.dual_aux_cooperative_lora import DualAuxCooperativeVLMWrapper
wrapper = DualAuxCooperativeVLMWrapper(
    model, lora_r=128, lora_alpha=256, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    num_comm_rounds=2, aux_a_hard_route=0.9, aux_b_hard_route=0.1,
)
wrapper.load_cooperative(sft_ckpt)
wrapper.eval()

# Load 3 sample episodes
with open("v12_gui_360/data/gui360_train_2000_balanced.jsonl") as f:
    episodes = [json.loads(f.readline()) for _ in range(3)]

for ep_i, ep in enumerate(episodes):
    step = ep["steps"][0]
    goal = ep["goal"]
    img_path = step["screenshot"]

    prompt = f"""You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{goal}

The history of actions are:
None

The actions supported are:
<action>
- click
  - Args: coordinate: [x, y], button: str, double: bool, pressed: str|None
- type
  - Args: coordinate: [x, y], keys: str, clear_current_text: bool, control_focus: bool
- drag
  - Args: start_coordinate: [x, y], end_coordinate: [x, y], button: str, duration: float
- wheel_mouse_input
  - Args: coordinate: [x, y], wheel_dist: int
</action>
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen.

In <aux_a></aux_a>, analyze from perspective A.
In <aux_b></aux_b>, analyze from perspective B, considering perspective A's analysis.
Then output your action inside <decision></decision> tags, with a <tool_call> block:
<decision>
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>
</decision>

Only **ONE** action should be taken at a time.

<aux_a>
"""

    messages = [{"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}

    print(f"\n{'='*70}")
    print(f"Episode {ep_i}: {goal[:80]}")
    print(f"Image: {img_path}")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"GT action: {step['action']}")

    # Test 1: Standard generation (no phase routing)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"\n--- GREEDY (no routing, 400 tok max) ---")
        print(repr(resp[:600]))
        print(f"Tokens: {out[0].shape[0] - inputs['input_ids'].shape[1]}")

    # Test 2: With temperature sampling (like training)
    with torch.no_grad():
        out2 = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=1.0, top_p=0.95)
        resp2 = processor.decode(out2[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"\n--- SAMPLED (temp=1.0, top_p=0.95, 400 tok max) ---")
        print(repr(resp2[:600]))
        print(f"Tokens: {out2[0].shape[0] - inputs['input_ids'].shape[1]}")

    # Test 3: With phase 0 routing set globally
    wrapper.set_phase_mask_global(0)
    with torch.no_grad():
        out3 = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        resp3 = processor.decode(out3[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"\n--- GREEDY + PHASE 0 (aux_a route=0.9) ---")
        print(repr(resp3[:600]))
        print(f"Tokens: {out3[0].shape[0] - inputs['input_ids'].shape[1]}")
    wrapper.clear_phase_mask()

print("\n=== DONE ===")
PYEOF
