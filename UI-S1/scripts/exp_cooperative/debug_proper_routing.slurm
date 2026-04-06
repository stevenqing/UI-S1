#!/bin/bash
#SBATCH --job-name=debug_coop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/debug_proper_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/debug_proper_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
cd "$PROJECT_DIR"

python -c "
import json, os, re, sys, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper

print('=== Loading model ===')
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'checkpoints/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cuda:0')
coop_ckpt = 'train_GUI_360/llamafactory/output/cooperative_thought_v1/final'
config = json.load(open(os.path.join(coop_ckpt, 'cooperative_config.json')))
lora_v = torch.load(os.path.join(coop_ckpt, 'lora_v.pt'), map_location='cpu', weights_only=True)
r = [v for k, v in lora_v.items() if 'lora_A_v' in k][0].shape[0]
model = CooperativeVLMWrapper(base_model, lora_r=r, lora_alpha=r*2, lora_dropout=0.0,
                               target_modules=config['target_modules'], bind_weight=0.0)
model.load_cooperative_checkpoint(coop_ckpt)
model.eval()
processor = AutoProcessor.from_pretrained('checkpoints/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)

# Collect 5 grounding samples
samples = []
data_dir = 'datasets/GUI-360/test/data/excel'
image_dir = 'datasets/GUI-360/test/image'
for cat in sorted(os.listdir(data_dir))[:2]:
    cat_dir = os.path.join(data_dir, cat)
    for status in sorted(os.listdir(cat_dir))[:1]:
        full_dir = os.path.join(cat_dir, status)
        for jf in sorted(os.listdir(full_dir)):
            if not jf.endswith('.jsonl'): continue
            with open(os.path.join(full_dir, jf)) as f:
                for line in f:
                    s = json.loads(line.strip())
                    if 'grounding' in s.get('step',{}).get('tags',[]):
                        s['_domain'] = 'excel'
                        s['_category'] = cat
                        samples.append(s)
            if len(samples) >= 5: break
        if len(samples) >= 5: break
    if len(samples) >= 5: break

print(f'Found {len(samples)} grounding samples')

for i, sample in enumerate(samples[:5]):
    step = sample['step']
    action = step['action']
    gt_x = action.get('coordinate_x')
    gt_y = action.get('coordinate_y')
    control_text = action.get('control_test', '')
    control_label = action.get('control_label', '')
    screenshot = step.get('screenshot_clean', '')
    img_path = os.path.join(image_dir, 'excel', sample['_category'], screenshot)

    print(f'\n=== Sample {i} ===')
    print(f'Image: {img_path}')
    print(f'Exists: {os.path.exists(img_path)}')
    print(f'GT: ({gt_x}, {gt_y})')
    print(f'Control: {control_text} (label: {control_label})')

    # Build prompt
    prompt = f'In this screenshot, please locate the UI element: {control_text}'
    if control_label:
        prompt += f' (label: {control_label})'

    image = Image.open(img_path).convert('RGB')
    content = [{'type': 'image', 'image': img_path}, {'type': 'text', 'text': prompt}]
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True)
    inputs = {k: v.to(base_model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    n_img = (inputs['input_ids'] == 151655).sum().item()
    print(f'IMAGE_PAD tokens: {n_img}')

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    print(f'RESPONSE: {response[:500]}')

    coord_match = re.search(r'\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', response)
    if coord_match:
        px, py = int(coord_match.group(1)), int(coord_match.group(2))
        hit = abs(px - gt_x) <= 14 and abs(py - gt_y) <= 14
        print(f'Parsed: ({px}, {py}), Hit: {hit}')
    else:
        print('NO COORDINATE PARSED')

    # Also test with base model only (no LoRA routing)
    model._set_token_mask(None)
    with torch.no_grad():
        output_ids_base = base_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response_base = processor.decode(output_ids_base[0][input_len:], skip_special_tokens=True)
    print(f'BASE RESPONSE: {response_base[:500]}')

    coord_match_b = re.search(r'\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', response_base)
    if coord_match_b:
        px, py = int(coord_match_b.group(1)), int(coord_match_b.group(2))
        hit = abs(px - gt_x) <= 14 and abs(py - gt_y) <= 14
        print(f'Base parsed: ({px}, {py}), Hit: {hit}')
    else:
        print('BASE: NO COORDINATE PARSED')

print('\n=== DONE ===')
"
