"""
Gate analysis v4: Reasoning Path Analysis.

Key hypothesis: Gates may encode different "reasoning paths" during generation.
During autoregressive decoding, the model generates tokens like:
  "I will click on the [element]...<action>{"action":"click","coordinate":[x,y]}</action>"

Different parts of this generation may use different gate patterns:
  - Planning phase ("I will click on...")
  - Target identification ("the search button")
  - Action formulation ("<action>{...}")
  - Coordinate generation ("[x, y]")

This experiment captures per-token gate values DURING GENERATION to see if
gates show distinct patterns at different reasoning stages.

Additionally: for the same input, we run multiple times with routing noise
to see if different routing leads to different gate patterns → different reasoning.
"""

import json
import os
import sys
from collections import defaultdict
import statistics

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import argparse
import re


class GenerationGateCollector:
    """Collect gate values per generated token during autoregressive decoding."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Per generation step: list of dicts {layer: {g12, g21, r}}
        # Only stores the LAST token's gate (the one being generated)
        self.gen_steps = []
        self._current_step = {}

    def start_step(self):
        """Call before each forward pass during generation."""
        self._current_step = {}

    def end_step(self):
        """Call after each forward pass during generation."""
        self.gen_steps.append(self._current_step)
        self._current_step = {}

    def hook_model(self, wrapper):
        from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear
        for name, module in wrapper.named_modules():
            if isinstance(module, IterativeCooperativeLoRALinear):
                parts = name.split('.')
                layer_idx = None
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try: layer_idx = int(parts[i + 1])
                        except: pass
                proj = parts[-1]
                self._patch(module, layer_idx, proj)

    def _patch(self, module, layer_idx, proj_name):
        collector = self

        def hooked_forward(x):
            base_out = module.base_linear(x)
            if module._route_weight is None:
                module._last_routing_weights = None
                return base_out
            x_drop = module.lora_dropout(x)
            dtype = x_drop.dtype
            w = module._route_weight.to(dtype)
            logits = F.linear(x_drop, w.unsqueeze(0))
            r = torch.sigmoid(logits)
            module._last_routing_weights = r.detach()

            h_1 = F.linear(x_drop, module.lora_A_1.to(dtype))
            h_2 = F.linear(x_drop, module.lora_A_2.to(dtype))

            if module._comm_params is not None:
                T = module._comm_params['T']
                for t in range(T):
                    g_12 = torch.sigmoid(
                        F.linear(h_1, module._comm_params['gate_12'][t].to(dtype).unsqueeze(0))
                    )
                    h_1 = h_1 + g_12 * F.linear(h_2, module._comm_params['W_12'][t].to(dtype))
                    g_21 = torch.sigmoid(
                        F.linear(h_2, module._comm_params['gate_21'][t].to(dtype).unsqueeze(0))
                    )
                    h_2 = h_2 + g_21 * F.linear(h_1, module._comm_params['W_21'][t].to(dtype))

                    # Only record for q_proj, round 1, last token
                    if proj_name == 'q_proj' and t == 1:
                        key = f"L{layer_idx:02d}"
                        collector._current_step[key] = {
                            'g12': g_12[0, -1, 0].item(),
                            'g21': g_21[0, -1, 0].item(),
                            'r': r[0, -1, 0].item(),
                        }

            h_blend = r * h_1 + (1 - r) * h_2
            delta = F.linear(h_blend, module.lora_B.to(dtype)) * module.scaling
            return base_out + delta
        module.forward = hooked_forward


SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a screenshot and a task instruction. "
    "Perform the next action to complete the task.\n\n"
    "Action space:\n"
    '  click: {"action": "click", "coordinate": [x, y]}\n'
    '  type: {"action": "type", "text": "content"}\n'
    '  drag: {"action": "drag", "coordinate": [x1, y1], "endCoordinate": [x2, y2]}\n'
    '  terminate: {"action": "terminate", "status": "success|failure"}\n\n'
    "Output format: <action>{JSON action}</action>"
)


def segment_generation(tokens, tokenizer):
    """Segment generated tokens into reasoning phases.
    Returns a list of phase strings, one per token.
    Phases: 'planning', 'action_start', 'action_type', 'coordinate'
    """
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Find <action> tag position
    action_start = text.find('<action>')
    if action_start == -1:
        # No action tag found — all tokens are planning
        return ['planning'] * len(tokens)

    # Map character position back to token positions
    segments = []
    cumlen = 0
    phase = 'planning'  # before <action>

    for i, tok_id in enumerate(tokens):
        tok_text = tokenizer.decode([tok_id])
        cumlen += len(tok_text)

        if cumlen <= action_start:
            phase = 'planning'
        else:
            # We're past the <action> tag
            decoded_so_far = tokenizer.decode(tokens[:i+1], skip_special_tokens=True)
            if '"coordinate"' in decoded_so_far and '[' in decoded_so_far.split('"coordinate"')[-1]:
                phase = 'coordinate'
            elif '"action"' in decoded_so_far:
                phase = 'action_type'
            else:
                phase = 'action_start'

        segments.append(phase)

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of generation runs per episode (with different noise)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Dir to save per-shard jsonl results")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    with open(args.test_data) as f:
        all_episodes = [json.loads(l) for l in f]
    all_episodes = all_episodes[:args.max_episodes]

    # Shard episodes across GPUs
    episodes = all_episodes[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(episodes)} episodes "
          f"(total {len(all_episodes)})")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": args.gpu},
    )
    from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    model = IterativeCooperativeVLMWrapper(
        base_model,
        lora_r=coop_config.get("lora_r", 128),
        lora_alpha=coop_config.get("lora_alpha", 256),
        target_modules=coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        num_comm_rounds=coop_config.get("num_comm_rounds", 2),
    )
    model.load_cooperative(args.coop_checkpoint)
    model.eval()

    collector = GenerationGateCollector()
    collector.hook_model(model)
    print("Hooks installed")

    focus_layers = ['L10', 'L18', 'L27']
    all_results = []

    for ep_idx, ep in enumerate(episodes):
        eid = ep['episode_id']
        goal = ep['goal']
        step = ep['steps'][0]
        screenshot = step['screenshot']

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        content = []
        if os.path.exists(screenshot):
            content.append({"type": "image", "image": f"file://{screenshot}"})
        content.append({"type": "text", "text": f"Task: {goal}"})
        messages.append({"role": "user", "content": content})

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if os.path.exists(screenshot):
                img = Image.open(screenshot).convert("RGB")
                inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

            # Run multiple generations to see if different noise → different reasoning
            run_results = []
            for run_idx in range(args.num_runs):
                collector.reset()
                # Use a continuous hook: each forward call = one "step"
                # model.generate() internally calls forward once for prefill,
                # then once per generated token
                collector.start_step()  # will be ended by first forward

                # Patch forward to auto-collect per-call
                _orig_start = collector.start_step
                _orig_end = collector.end_step
                _call_count = [0]

                def _auto_hook(*args, **kwargs):
                    if _call_count[0] > 0:
                        _orig_end()
                        _orig_start()
                    _call_count[0] += 1

                # Install a pre-forward hook on the base model to track each forward call
                _hook_handle = model.base_model.register_forward_pre_hook(
                    lambda mod, inp: _auto_hook()
                )

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    gen_kwargs = {k: v for k, v in inputs.items()}
                    gen_kwargs["max_new_tokens"] = 256
                    gen_kwargs["do_sample"] = False
                    output_ids = model.generate(**gen_kwargs)
                    # Get only generated tokens (strip prompt)
                    prompt_len = inputs["input_ids"].shape[1]
                    generated_ids = output_ids[0, prompt_len:].tolist()

                _hook_handle.remove()
                collector.end_step()  # end the last step

                # Segment the generation into phases
                gen_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                segments = segment_generation(generated_ids, processor.tokenizer)

                # Collect gate values per phase (skip prefill step [0])
                phase_gates = defaultdict(lambda: defaultdict(list))
                for tok_idx, phase in enumerate(segments):
                    step_idx = tok_idx + 1  # +1 because step 0 is prefill
                    if step_idx < len(collector.gen_steps):
                        step_data = collector.gen_steps[step_idx]
                        for layer in focus_layers:
                            if layer in step_data:
                                g_avg = (step_data[layer]['g12'] + step_data[layer]['g21']) / 2
                                phase_gates[phase][layer].append(g_avg)

                run_result = {
                    'text': gen_text,
                    'num_tokens': len(generated_ids),
                    'phases': {},
                }
                for phase, layer_data in phase_gates.items():
                    run_result['phases'][phase] = {}
                    for layer, vals in layer_data.items():
                        run_result['phases'][phase][layer] = {
                            'mean': statistics.mean(vals) if vals else None,
                            'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                            'n': len(vals),
                        }

                # Also capture the raw gate trajectory for focus layers
                gate_trajectory = {layer: [] for layer in focus_layers}
                for step_idx in range(1, len(collector.gen_steps)):
                    step_data = collector.gen_steps[step_idx]
                    for layer in focus_layers:
                        if layer in step_data:
                            g_avg = (step_data[layer]['g12'] + step_data[layer]['g21']) / 2
                            gate_trajectory[layer].append(g_avg)

                run_result['trajectory'] = gate_trajectory
                run_results.append(run_result)

            all_results.append({
                'eid': eid,
                'goal': goal[:80],
                'runs': run_results,
            })

        except Exception as e:
            import traceback
            print(f"Error ep{eid}: {e}")
            traceback.print_exc()
            continue

        print(f"  [{ep_idx+1}/{len(episodes)}] ep{eid}: {len(run_results)} runs, "
              f"tokens={run_results[0]['num_tokens'] if run_results else 0}")

    # ── Save per-shard results ──
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        shard_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
        with open(shard_file, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')
        print(f"Saved {len(all_results)} results to {shard_file}")

        # If this is not the only shard, skip analysis (will be done by merge step)
        if args.num_shards > 1:
            print(f"Shard {args.shard_id} done. Run merge separately for full analysis.")
            return

    # ── Analysis ──
    print(f"\nCollected {len(all_results)} episodes × {args.num_runs} runs")

    # 1. Gate values by reasoning phase
    print("\n" + "=" * 80)
    print("GATE VALUES BY REASONING PHASE (across all episodes)")
    print("=" * 80)

    global_phase_gates = defaultdict(lambda: defaultdict(list))
    for ep in all_results:
        for run in ep['runs']:
            for phase, layers in run['phases'].items():
                for layer, stats in layers.items():
                    if stats['mean'] is not None:
                        global_phase_gates[phase][layer].append(stats['mean'])

    for phase in ['planning', 'action_start', 'action_type', 'coordinate']:
        if phase in global_phase_gates:
            print(f"\n  Phase: {phase}")
            for layer in focus_layers:
                vals = global_phase_gates[phase][layer]
                if vals:
                    print(f"    {layer}: gate={statistics.mean(vals):.4f} ± {statistics.stdev(vals) if len(vals)>1 else 0:.4f}, n={len(vals)}")

    # 2. Gate trajectory variance (does gate change during generation?)
    print("\n" + "=" * 80)
    print("GATE TRAJECTORY VARIANCE (within single generation)")
    print("=" * 80)

    for layer in focus_layers:
        within_variances = []
        for ep in all_results:
            for run in ep['runs']:
                traj = run['trajectory'].get(layer, [])
                if len(traj) > 3:
                    within_variances.append(statistics.stdev(traj))
        if within_variances:
            print(f"  {layer}: avg within-generation std = {statistics.mean(within_variances):.4f} "
                  f"(range: {min(within_variances):.4f} ~ {max(within_variances):.4f})")

    # 3. Cross-run consistency (same input, different outputs?)
    print("\n" + "=" * 80)
    print("CROSS-RUN ANALYSIS (same input, multiple generations)")
    print("=" * 80)

    same_output_count = 0
    diff_output_count = 0
    gate_diff_when_same = []
    gate_diff_when_diff = []

    for ep in all_results:
        runs = ep['runs']
        if len(runs) < 2:
            continue

        # Compare outputs
        texts = [r['text'] for r in runs]
        all_same = all(t == texts[0] for t in texts)

        if all_same:
            same_output_count += 1
        else:
            diff_output_count += 1

        # Compare gate trajectories
        for layer in focus_layers:
            trajs = [r['trajectory'].get(layer, []) for r in runs]
            if all(len(t) > 0 for t in trajs):
                # Mean gate per run
                means = [statistics.mean(t) for t in trajs]
                gate_spread = max(means) - min(means)
                if all_same:
                    gate_diff_when_same.append(gate_spread)
                else:
                    gate_diff_when_diff.append(gate_spread)

    print(f"  Same output across runs: {same_output_count}/{same_output_count+diff_output_count}")
    print(f"  Different output across runs: {diff_output_count}/{same_output_count+diff_output_count}")
    if gate_diff_when_same:
        print(f"  Gate spread when same output: {statistics.mean(gate_diff_when_same):.4f}")
    if gate_diff_when_diff:
        print(f"  Gate spread when diff output: {statistics.mean(gate_diff_when_diff):.4f}")

    # 4. Show example gate trajectories
    print("\n" + "=" * 80)
    print("EXAMPLE GATE TRAJECTORIES (first 3 episodes)")
    print("=" * 80)

    for ep in all_results[:3]:
        print(f"\n  Episode {ep['eid']}: {ep['goal']}")
        for run_idx, run in enumerate(ep['runs'][:1]):  # Just first run
            print(f"    Generated: {run['text'][:100]}...")
            for layer in focus_layers:
                traj = run['trajectory'].get(layer, [])
                if len(traj) > 5:
                    # Show first 5, middle, last 5
                    first = [f"{v:.3f}" for v in traj[:5]]
                    last = [f"{v:.3f}" for v in traj[-5:]]
                    print(f"    {layer}: [{', '.join(first)}, ..., {', '.join(last)}] (len={len(traj)})")


def merge_and_analyze(output_dir):
    """Load all shard files and run the analysis."""
    import glob as globmod
    shard_files = sorted(globmod.glob(os.path.join(output_dir, "shard_*.jsonl")))
    print(f"Found {len(shard_files)} shard files in {output_dir}")

    all_results = []
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                all_results.append(json.loads(line))
    print(f"Total: {len(all_results)} episodes")

    focus_layers = ['L10', 'L18', 'L27']

    # 1. Gate values by reasoning phase
    print("\n" + "=" * 80)
    print("GATE VALUES BY REASONING PHASE (across all episodes)")
    print("=" * 80)

    global_phase_gates = defaultdict(lambda: defaultdict(list))
    for ep in all_results:
        for run in ep['runs']:
            for phase, layers in run['phases'].items():
                for layer, stats in layers.items():
                    if stats['mean'] is not None:
                        global_phase_gates[phase][layer].append(stats['mean'])

    for phase in ['planning', 'action_start', 'action_type', 'coordinate']:
        if phase in global_phase_gates:
            print(f"\n  Phase: {phase}")
            for layer in focus_layers:
                vals = global_phase_gates[phase][layer]
                if vals:
                    print(f"    {layer}: gate={statistics.mean(vals):.4f} ± "
                          f"{statistics.stdev(vals) if len(vals)>1 else 0:.4f}, n={len(vals)}")

    # 2. Gate trajectory variance
    print("\n" + "=" * 80)
    print("GATE TRAJECTORY VARIANCE (within single generation)")
    print("=" * 80)

    for layer in focus_layers:
        within_variances = []
        for ep in all_results:
            for run in ep['runs']:
                traj = run['trajectory'].get(layer, [])
                if len(traj) > 3:
                    within_variances.append(statistics.stdev(traj))
        if within_variances:
            print(f"  {layer}: avg within-generation std = {statistics.mean(within_variances):.4f} "
                  f"(range: {min(within_variances):.4f} ~ {max(within_variances):.4f})")

    # 3. Phase transition: gate change between phases
    print("\n" + "=" * 80)
    print("PHASE TRANSITION: Planning → Action")
    print("=" * 80)

    for layer in focus_layers:
        planning_vals = global_phase_gates.get('planning', {}).get(layer, [])
        coord_vals = global_phase_gates.get('coordinate', {}).get(layer, [])
        atype_vals = global_phase_gates.get('action_type', {}).get(layer, [])
        if planning_vals and coord_vals:
            diff = statistics.mean(coord_vals) - statistics.mean(planning_vals)
            print(f"  {layer}: planning={statistics.mean(planning_vals):.4f}, "
                  f"action_type={statistics.mean(atype_vals):.4f if atype_vals else 'N/A'}, "
                  f"coordinate={statistics.mean(coord_vals):.4f}, "
                  f"Δ(coord-plan)={diff:+.4f}")

    # 4. Example trajectories
    print("\n" + "=" * 80)
    print("EXAMPLE GATE TRAJECTORIES (first 5 episodes)")
    print("=" * 80)

    for ep in all_results[:5]:
        print(f"\n  Episode {ep['eid']}: {ep['goal']}")
        for run in ep['runs'][:1]:
            print(f"    Generated: {run['text'][:120]}...")
            for layer in focus_layers:
                traj = run['trajectory'].get(layer, [])
                if len(traj) > 5:
                    first = [f"{v:.3f}" for v in traj[:5]]
                    last = [f"{v:.3f}" for v in traj[-5:]]
                    print(f"    {layer}: [{', '.join(first)}, ..., {', '.join(last)}] (len={len(traj)})")

    # 5. Save summary
    summary_file = os.path.join(output_dir, "analysis_summary.json")
    summary = {}
    for phase in ['planning', 'action_start', 'action_type', 'coordinate']:
        if phase in global_phase_gates:
            summary[phase] = {}
            for layer in focus_layers:
                vals = global_phase_gates[phase][layer]
                if vals:
                    summary[phase][layer] = {
                        'mean': statistics.mean(vals),
                        'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                        'n': len(vals),
                    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--merge':
        merge_and_analyze(sys.argv[2])
    else:
        main()
