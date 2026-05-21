"""Detailed hidden state analysis for learned router design.

Instead of forcing a binary WHERE/WHAT split, this script:
1. Extracts hidden states for 5 token types: IMAGE, COORD, THINK, ACTION, OTHER
2. Sub-divides THINK by structural sections (description, elements, status, decision)
3. Computes affinity spectrum: for each think token, its cosine sim to IMAGE mean vs ACTION mean
4. Runs PCA to visualize token clusters per layer
5. Tests linear separability with varying groupings

Usage:
    python evaluation/verify_hidden_state_detailed.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --data datasets/cooperative_thought_ac/ac_train_thought.jsonl \
        --n_samples 15 \
        --layers 0,7,14,21,27
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Token constants (from cooperative_wrapper.py)
# ═══════════════════════════════════════════════════════════════════════

IMAGE_PAD_ID = 151655

# Coord routing
COORD_KEY_ID = 62526
BBOX_KEY_ID = 58456
DIGIT_TOKEN_IDS = set(range(15, 25))
COORD_PUNCT_IDS = {13, 11, 220}
COORD_ALL_VALUE_IDS = DIGIT_TOKEN_IDS | COORD_PUNCT_IDS
BRACKET_OPEN_ID = 508
BRACKET_CLOSE_IDS = {1125, 81136}
COORD_KEY_TRAIL_IDS = {788, 1, 330} | DIGIT_TOKEN_IDS

# Think/thought trigrams
THINK_OPEN_TRIGRAM = (13708, 766, 29)     # <think>
THINK_CLOSE_TRIGRAM = (522, 26865, 29)    # </think>
THOUGHT_OPEN_TRIGRAM = (13708, 2450, 29)  # <thought>
THOUGHT_CLOSE_TRIGRAM = (522, 60565, 29)  # </thought>

# Chat template
IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
NEWLINE_ID = 198


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_samples(jsonl_path: str, n_samples: int) -> List[dict]:
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if not item.get("has_thought", False):
                continue
            if not item.get("images"):
                continue
            samples.append(item)
            if len(samples) >= n_samples:
                break
    print(f"Loaded {len(samples)} samples from {jsonl_path}")
    return samples


def sample_to_qwen_messages(sample: dict) -> Tuple[list, list]:
    conversations = sample["conversations"]
    images = sample.get("images", [])
    qwen_messages = []
    pil_images = []
    image_idx = 0
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        content_blocks = []
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if i > 0 and image_idx < len(images):
                img_path = images[image_idx]
                if os.path.exists(img_path):
                    pil = Image.open(img_path).convert("RGB")
                    pil_images.append(pil)
                    content_blocks.append({"type": "image", "image": pil})
                image_idx += 1
            if part.strip():
                content_blocks.append({"type": "text", "text": part})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": text})
        qwen_messages.append({"role": role, "content": content_blocks})
    return qwen_messages, pil_images


# ═══════════════════════════════════════════════════════════════════════
# Token classification
# ═══════════════════════════════════════════════════════════════════════

def _match_tri(ids, i, tri):
    return (i + 2 < len(ids)
            and ids[i] == tri[0] and ids[i+1] == tri[1] and ids[i+2] == tri[2])


def find_think_spans(input_ids: List[int]) -> List[Tuple[int, int]]:
    """Find (start, end) inclusive spans for <think>...</think>."""
    spans = []
    i = 0
    n = len(input_ids)
    while i < n - 2:
        if _match_tri(input_ids, i, THINK_OPEN_TRIGRAM) or \
           _match_tri(input_ids, i, THOUGHT_OPEN_TRIGRAM):
            start = i
            j = i + 3
            while j < n - 2:
                if _match_tri(input_ids, j, THINK_CLOSE_TRIGRAM) or \
                   _match_tri(input_ids, j, THOUGHT_CLOSE_TRIGRAM):
                    spans.append((start, j + 2))
                    i = j + 3
                    break
                j += 1
            else:
                spans.append((start, n - 1))
                break
            continue
        i += 1
    return spans


def find_coord_indices(input_ids: List[int]) -> set:
    """Find coordinate/bbox digit token indices (mirrors _mark_coord_tokens)."""
    indices = set()
    n = len(input_ids)
    in_coord = False
    in_bracket = False
    for i in range(n):
        tid = input_ids[i]
        if not in_coord:
            if tid == COORD_KEY_ID or tid == BBOX_KEY_ID:
                in_coord = True
                in_bracket = False
        else:
            if not in_bracket:
                if tid == BRACKET_OPEN_ID:
                    in_bracket = True
                elif tid not in COORD_KEY_TRAIL_IDS:
                    in_coord = False
            else:
                if tid in BRACKET_CLOSE_IDS:
                    in_coord = False
                    in_bracket = False
                elif tid in COORD_ALL_VALUE_IDS:
                    indices.add(i)
                else:
                    in_coord = False
                    in_bracket = False
    return indices


def find_assistant_spans(input_ids: List[int]) -> List[Tuple[int, int]]:
    """Find assistant response spans from chat template markers."""
    spans = []
    n = len(input_ids)
    i = 0
    while i < n - 2:
        if (input_ids[i] == IM_START_ID
                and input_ids[i+1] == ASSISTANT_ID
                and input_ids[i+2] == NEWLINE_ID):
            start = i + 3  # skip header
            j = start
            while j < n:
                if input_ids[j] == IM_END_ID:
                    spans.append((start, j - 1))
                    i = j + 1
                    break
                j += 1
            else:
                spans.append((start, n - 1))
                break
            continue
        i += 1
    return spans


def segment_think_by_structure(input_ids: List[int], think_start: int,
                                think_end: int, tokenizer) -> Dict[str, List[int]]:
    """Split a think span into structural sections based on numbered patterns.

    AC think text typically follows:
      1. **App and Screen/Page**: ...  (DESCRIPTION)
      2. **Main UI Elements**: ...     (ELEMENTS)
      3. **Text Content**: ...         (CONTENT)
      4. **Active States**: ...        (STATUS)
      5. **Suggested Action**: ...     (DECISION)

    We detect section boundaries by looking for "N. **" or "N." patterns.
    Last section (action/suggestion) is classified as DECISION, others as DESCRIPTION.
    """
    # Skip the <think> trigram itself
    content_start = think_start + 3
    content_end = think_end - 3  # skip </think> trigram
    if content_end <= content_start:
        return {"THINK_DESC": list(range(content_start, think_end + 1)),
                "THINK_DECIDE": []}

    # Decode to find section boundaries
    think_ids = input_ids[content_start:content_end + 1]
    text = tokenizer.decode(think_ids, skip_special_tokens=False)

    # Find section starts: "N. **" or just "N." at line beginning
    section_pattern = re.compile(r'\n\s*(\d+)\.\s')
    sections = []
    for m in section_pattern.finditer(text):
        sections.append(m.start())

    if not sections:
        # No clear structure - all as DESCRIPTION
        return {"THINK_DESC": list(range(content_start, content_end + 1)),
                "THINK_DECIDE": []}

    # Map character positions to token indices
    char_to_tok = {}
    cum_chars = 0
    for ti, tid in enumerate(think_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        for c in range(len(tok_text)):
            char_to_tok[cum_chars + c] = ti
        cum_chars += len(tok_text)

    # Find the last section boundary
    last_section_char = sections[-1]
    last_section_tok = char_to_tok.get(last_section_char, len(think_ids) - 1)

    desc_indices = []
    decide_indices = []
    for ti in range(len(think_ids)):
        abs_idx = content_start + ti
        if ti >= last_section_tok:
            decide_indices.append(abs_idx)
        else:
            desc_indices.append(abs_idx)

    return {"THINK_DESC": desc_indices, "THINK_DECIDE": decide_indices}


def classify_tokens(input_ids: List[int], tokenizer) -> Dict[str, List[int]]:
    """Classify tokens into: IMAGE, COORD, THINK_DESC, THINK_DECIDE, ACTION, OTHER."""
    n = len(input_ids)

    image_indices = {i for i in range(n) if input_ids[i] == IMAGE_PAD_ID}
    coord_indices = find_coord_indices(input_ids) - image_indices

    think_spans = find_think_spans(input_ids)
    think_desc = []
    think_decide = []
    think_all_indices = set()
    for start, end in think_spans:
        segments = segment_think_by_structure(input_ids, start, end, tokenizer)
        for idx in segments["THINK_DESC"]:
            if idx not in image_indices and idx not in coord_indices:
                think_desc.append(idx)
                think_all_indices.add(idx)
        for idx in segments["THINK_DECIDE"]:
            if idx not in image_indices and idx not in coord_indices:
                think_decide.append(idx)
                think_all_indices.add(idx)

    # Assistant tokens that are not think and not coord
    assistant_spans = find_assistant_spans(input_ids)
    action_indices = []
    for start, end in assistant_spans:
        for i in range(start, end + 1):
            if (i not in image_indices and i not in coord_indices
                    and i not in think_all_indices):
                action_indices.append(i)

    # Everything else
    classified = image_indices | coord_indices | think_all_indices | set(action_indices)
    other_indices = [i for i in range(n) if i not in classified]

    return {
        "IMAGE": sorted(image_indices),
        "COORD": sorted(coord_indices),
        "THINK_DESC": think_desc,
        "THINK_DECIDE": think_decide,
        "ACTION": action_indices,
        "OTHER": other_indices,
    }


# ═══════════════════════════════════════════════════════════════════════
# Hidden state extraction
# ═══════════════════════════════════════════════════════════════════════

class HiddenStateCollector:
    def __init__(self, model, layer_indices: List[int]):
        self.layer_indices = layer_indices
        self.hidden_states = {}
        self._hooks = []

        # Find transformer layers
        vlm = model.model
        if hasattr(vlm, "language_model"):
            layers = vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            layers = vlm.layers
        else:
            raise AttributeError(f"Cannot find layers in {type(vlm).__name__}")

        for li in layer_indices:
            layer = layers[li]

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output[0] is the hidden state leaving this layer
                    h = output[0] if isinstance(output, tuple) else output
                    self.hidden_states[layer_idx] = h.detach().float().cpu()
                return hook_fn

            handle = layer.register_forward_hook(make_hook(li))
            self._hooks.append(handle)

    def clear(self):
        self.hidden_states.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def cosine_sim_vectors(v1, v2):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(
        torch.tensor(v1).unsqueeze(0),
        torch.tensor(v2).unsqueeze(0)
    ).item()


def compute_affinity_spectrum(think_vectors, image_mean, action_mean):
    """For each think token, compute its position on the IMAGE↔ACTION spectrum.

    Returns array of values in [-1, 1]:
      +1 = closer to IMAGE, -1 = closer to ACTION, 0 = equidistant
    """
    if len(think_vectors) == 0:
        return np.array([])

    image_mean_t = torch.tensor(image_mean).unsqueeze(0)   # [1, D]
    action_mean_t = torch.tensor(action_mean).unsqueeze(0)  # [1, D]
    think_t = torch.tensor(np.array(think_vectors))          # [N, D]

    sim_to_image = F.cosine_similarity(think_t, image_mean_t).numpy()   # [N]
    sim_to_action = F.cosine_similarity(think_t, action_mean_t).numpy()  # [N]

    # Spectrum: positive = closer to image, negative = closer to action
    spectrum = sim_to_image - sim_to_action
    return spectrum, sim_to_image, sim_to_action


def linear_probe_accuracy(features, labels):
    """Simple logistic regression probe accuracy (numpy-only fallback)."""
    from numpy.linalg import lstsq

    n = len(labels)
    if n < 10:
        return float('nan'), n

    # Normalize features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    X = (features - mean) / std

    # Add bias
    X = np.hstack([X, np.ones((n, 1))])
    y = labels.astype(float)

    # Least squares
    w, _, _, _ = lstsq(X, y, rcond=None)
    preds = (X @ w > 0.5).astype(int)
    acc = (preds == labels).mean()
    return acc, n


def run_analysis(all_vectors, layer_idx):
    """Run full analysis for one layer."""
    types = ["IMAGE", "COORD", "THINK_DESC", "THINK_DECIDE", "ACTION"]
    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx}")
    print(f"{'='*70}")

    # Token counts
    for t in types:
        n = len(all_vectors.get(t, []))
        print(f"  {t:15s}: {n:6d} vectors", end="")
        if n > 0:
            norms = [np.linalg.norm(v) for v in all_vectors[t]]
            print(f"  (mean norm = {np.mean(norms):.1f})")
        else:
            print()

    # Compute means
    means = {}
    for t in types:
        vecs = all_vectors.get(t, [])
        if len(vecs) > 0:
            means[t] = np.mean(vecs, axis=0)

    # Pairwise cosine similarity
    print(f"\n  Pairwise cosine similarity:")
    available = [t for t in types if t in means]
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            t1, t2 = available[i], available[j]
            sim = cosine_sim_vectors(means[t1], means[t2])
            print(f"    {t1:15s} vs {t2:15s}: {sim:+.4f}")

    # Affinity spectrum for THINK tokens
    if "IMAGE" in means and "ACTION" in means:
        for think_type in ["THINK_DESC", "THINK_DECIDE"]:
            vecs = all_vectors.get(think_type, [])
            if len(vecs) > 0:
                spectrum, sim_img, sim_act = compute_affinity_spectrum(
                    vecs, means["IMAGE"], means["ACTION"])
                print(f"\n  {think_type} affinity spectrum (IMAGE↔ACTION):")
                print(f"    mean(sim_to_IMAGE) = {np.mean(sim_img):.4f}")
                print(f"    mean(sim_to_ACTION) = {np.mean(sim_act):.4f}")
                print(f"    spectrum mean = {np.mean(spectrum):+.4f} "
                      f"({'closer to IMAGE' if np.mean(spectrum) > 0 else 'closer to ACTION'})")
                print(f"    spectrum std = {np.std(spectrum):.4f}")
                # Histogram
                bins = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
                hist, _ = np.histogram(spectrum, bins=bins)
                total = len(spectrum)
                print(f"    distribution:")
                for k in range(len(bins) - 1):
                    pct = hist[k] / total * 100
                    bar = '#' * int(pct / 2)
                    print(f"      [{bins[k]:+.1f},{bins[k+1]:+.1f}): "
                          f"{hist[k]:5d} ({pct:5.1f}%) {bar}")

    # Linear probe: multiple groupings
    print(f"\n  Linear probe (logistic regression):")

    # Grouping 1: IMAGE vs everything else
    if "IMAGE" in all_vectors and len(all_vectors["IMAGE"]) > 0:
        non_image = []
        for t in ["COORD", "THINK_DESC", "THINK_DECIDE", "ACTION"]:
            non_image.extend(all_vectors.get(t, []))
        if len(non_image) > 10:
            feats = np.array(all_vectors["IMAGE"][:2000] + non_image[:2000])
            labels = np.array([1]*min(len(all_vectors["IMAGE"]), 2000)
                             + [0]*min(len(non_image), 2000))
            acc, n = linear_probe_accuracy(feats, labels)
            print(f"    IMAGE vs rest:          acc={acc:.4f} (n={n})")

    # Grouping 2: THINK_DESC vs THINK_DECIDE
    desc_vecs = all_vectors.get("THINK_DESC", [])
    decide_vecs = all_vectors.get("THINK_DECIDE", [])
    if len(desc_vecs) > 10 and len(decide_vecs) > 10:
        feats = np.array(desc_vecs[:2000] + decide_vecs[:2000])
        labels = np.array([1]*min(len(desc_vecs), 2000)
                         + [0]*min(len(decide_vecs), 2000))
        acc, n = linear_probe_accuracy(feats, labels)
        print(f"    THINK_DESC vs DECIDE:   acc={acc:.4f} (n={n})")

    # Grouping 3: (IMAGE+COORD) vs (ACTION)
    where_vecs = all_vectors.get("IMAGE", [])[:1000] + all_vectors.get("COORD", [])
    what_vecs = all_vectors.get("ACTION", [])
    if len(where_vecs) > 10 and len(what_vecs) > 10:
        feats = np.array(where_vecs[:2000] + what_vecs[:2000])
        labels = np.array([1]*min(len(where_vecs), 2000)
                         + [0]*min(len(what_vecs), 2000))
        acc, n = linear_probe_accuracy(feats, labels)
        print(f"    IMAGE+COORD vs ACTION:  acc={acc:.4f} (n={n})")

    # Grouping 4: (IMAGE+COORD+THINK_DESC) vs (THINK_DECIDE+ACTION)
    where2 = (all_vectors.get("IMAGE", [])[:500]
              + all_vectors.get("COORD", [])
              + all_vectors.get("THINK_DESC", [])[:500])
    what2 = (all_vectors.get("THINK_DECIDE", [])
             + all_vectors.get("ACTION", []))
    if len(where2) > 10 and len(what2) > 10:
        feats = np.array(where2[:2000] + what2[:2000])
        labels = np.array([1]*min(len(where2), 2000)
                         + [0]*min(len(what2), 2000))
        acc, n = linear_probe_accuracy(feats, labels)
        print(f"    IMG+COORD+DESC vs DEC+ACT: acc={acc:.4f} (n={n})")

    # Grouping 5: THINK (all) vs ACTION
    think_all = desc_vecs + decide_vecs
    if len(think_all) > 10 and len(what_vecs) > 10:
        feats = np.array(think_all[:2000] + what_vecs[:2000])
        labels = np.array([1]*min(len(think_all), 2000)
                         + [0]*min(len(what_vecs), 2000))
        acc, n = linear_probe_accuracy(feats, labels)
        print(f"    THINK_ALL vs ACTION:    acc={acc:.4f} (n={n})")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n_samples", type=int, default=15)
    parser.add_argument("--layers", type=str, default="0,7,14,21,27")
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("Detailed Hidden State Analysis for Learned Router Design")
    print("=" * 70)
    print(f"Base model:  {args.base_model}")
    print(f"Data:        {args.data}")
    print(f"N samples:   {args.n_samples}")
    print(f"Layers:      {layer_indices}")

    # Load model
    print("\nLoading base model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"Device: {device}")

    # Setup hooks
    collector = HiddenStateCollector(model, layer_indices)

    # Load data
    samples = load_samples(args.data, args.n_samples)

    # Collect vectors per layer per type
    # Structure: {layer_idx: {type_name: [vectors]}}
    all_vectors = {li: {t: [] for t in
                        ["IMAGE", "COORD", "THINK_DESC", "THINK_DECIDE", "ACTION", "OTHER"]}
                   for li in layer_indices}

    print(f"\nProcessing {len(samples)} samples...")
    for si, sample in enumerate(samples):
        qwen_msgs, pil_images = sample_to_qwen_messages(sample)

        # Tokenize
        text = processor.apply_chat_template(qwen_msgs, tokenize=False,
                                             add_generation_prompt=False)
        inputs = processor(
            text=[text], images=pil_images if pil_images else None,
            return_tensors="pt", padding=True
        )
        input_ids = inputs["input_ids"][0].tolist()

        # Classify tokens
        token_types = classify_tokens(input_ids, processor.tokenizer)

        counts = {t: len(v) for t, v in token_types.items()}
        print(f"  Sample {si}: seq_len={len(input_ids)}, "
              + ", ".join(f"{t}={c}" for t, c in counts.items() if c > 0))

        # Forward pass
        collector.clear()
        with torch.no_grad():
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()
                          if isinstance(v, torch.Tensor)}
            model(**inputs_gpu)

        # Extract vectors per type per layer
        for li in layer_indices:
            h = collector.hidden_states.get(li)
            if h is None:
                continue
            h = h[0].numpy()  # [seq_len, hidden_dim], first batch element

            for type_name, indices in token_types.items():
                if not indices:
                    continue
                # Sample up to 200 vectors per type per sample to keep memory manageable
                sample_indices = indices[:200] if type_name == "IMAGE" else indices
                for idx in sample_indices:
                    if idx < h.shape[0]:
                        all_vectors[li][type_name].append(h[idx])

    collector.remove_hooks()

    # Run analysis per layer
    for li in layer_indices:
        run_analysis(all_vectors[li], li)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Layer':>5} | {'THINK_DESC→IMG':>14} | {'THINK_DESC→ACT':>14} | "
          f"{'THINK_DEC→IMG':>13} | {'THINK_DEC→ACT':>13} | "
          f"{'DESC vs DEC probe':>17}")
    print("-" * 90)

    for li in layer_indices:
        vecs = all_vectors[li]
        row = f"{li:>5} | "

        img_mean = np.mean(vecs["IMAGE"], axis=0) if vecs["IMAGE"] else None
        act_mean = np.mean(vecs["ACTION"], axis=0) if vecs["ACTION"] else None

        for think_type in ["THINK_DESC", "THINK_DECIDE"]:
            tvecs = vecs.get(think_type, [])
            if tvecs and img_mean is not None and act_mean is not None:
                _, sim_img, sim_act = compute_affinity_spectrum(tvecs, img_mean, act_mean)
                row += f"{np.mean(sim_img):>14.4f} | {np.mean(sim_act):>14.4f} | "
            else:
                row += f"{'N/A':>14} | {'N/A':>14} | "

        # DESC vs DECIDE probe
        if len(vecs.get("THINK_DESC", [])) > 10 and len(vecs.get("THINK_DECIDE", [])) > 10:
            feats = np.array(vecs["THINK_DESC"][:2000] + vecs["THINK_DECIDE"][:2000])
            labels = np.array([1]*min(len(vecs["THINK_DESC"]), 2000)
                             + [0]*min(len(vecs["THINK_DECIDE"]), 2000))
            acc, _ = linear_probe_accuracy(feats, labels)
            row += f"{acc:>17.4f}"
        else:
            row += f"{'N/A':>17}"

        print(row)

    print(f"\n{'='*70}")
    print("Key questions for router design:")
    print("  1. If THINK_DESC sim_to_IMAGE >> sim_to_ACTION → route desc to WHERE")
    print("  2. If THINK_DECIDE sim_to_ACTION >> sim_to_IMAGE → route decide to WHAT")
    print("  3. If DESC vs DECIDE probe acc >> 0.5 → linear router can split think")
    print("  4. How does this change across layers → per-layer or shared router?")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
