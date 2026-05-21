"""Verify whether Qwen2.5-VL hidden states naturally separate WHERE vs WHAT tokens.

This script probes the base model (no LoRA) to measure how well hidden-state
geometry already distinguishes spatial/visual tokens from reasoning/decision
tokens -- i.e., whether cooperative routing has a signal to exploit.

Token taxonomy:
  WHERE (spatial / visual):
    - IMAGE:         image pad tokens  (id == 151655)
    - COORD:         coordinate/bbox digit tokens inside action JSON
    - THINK_SPATIAL: think tokens describing visual layout (screen, button, ...)

  WHAT (reasoning / decision):
    - THINK_DECISION: think tokens about reasoning (need to, should, click, ...)
    - ACTION_TYPE:    non-coordinate action tokens (action name, keys, etc.)

For each layer in {0, 7, 14, 21, 27}:
  1. Mean hidden vector per token type
  2. Pairwise cosine similarity between type means
  3. Linear probe: WHERE vs WHAT classification accuracy (logistic regression)

Usage:
    python evaluation/verify_hidden_state_routing.py \
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
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Token ID constants (from cooperative_wrapper.py) ──────────────────
IMAGE_PAD_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653

COORD_KEY_ID = 62526       # "coordinate"
BBOX_KEY_ID = 58456        # "bbox"
DIGIT_TOKEN_IDS = set(range(15, 25))  # 0-9
COORD_PUNCT_IDS = {13, 11, 220}      # '.', ',', ' '
COORD_ALL_VALUE_IDS = DIGIT_TOKEN_IDS | COORD_PUNCT_IDS
BRACKET_OPEN_ID = 508      # ' ['
BRACKET_CLOSE_IDS = {1125, 81136}    # '],', ']}'
COORD_KEY_TRAIL_IDS = {788, 1, 330} | DIGIT_TOKEN_IDS  # '":' etc.

# Thought trigram tokens (AC data uses <think>, not <thought>)
# <think>   = (13708, 766, 29)   i.e. "<th" + "ink" + ">"
# </think>  = (522, 26865, 29)   i.e. "</" + "think" + ">"
# <thought> = (13708, 2450, 29)  i.e. "<th" + "ought" + ">"
# </thought>= (522, 60565, 29)   i.e. "</" + "thought" + ">"
THINK_OPEN_TRIGRAM = (13708, 766, 29)
THINK_CLOSE_TRIGRAM = (522, 26865, 29)
THOUGHT_OPEN_TRIGRAM = (13708, 2450, 29)
THOUGHT_CLOSE_TRIGRAM = (522, 60565, 29)

# Action bigram tokens  (we detect <action>...</action> similarly)
# <action> tokenizes as: 27 ("<") + 1765 ("action") + 29 (">")  -- but we use regex on text
# Instead, we detect the action span via the text content after decoding.

# Chat template special tokens
IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
NEWLINE_ID = 198

# ── Spatial/decision word lists for think-token classification ────────
SPATIAL_WORDS = {
    "screen", "button", "icon", "top", "bottom", "left", "right",
    "display", "image", "layout", "bar", "tab", "menu", "banner",
    "header", "footer", "corner", "center", "middle", "above",
    "below", "upper", "lower", "visible", "interface", "page",
    "section", "area", "position", "located", "appears",
}

DECISION_WORDS = {
    "need to", "should", "click", "next step", "action",
    "tap", "swipe", "type", "enter", "select", "open",
    "navigate", "scroll", "press", "input", "go to",
    "perform", "execute", "choose", "decide", "plan",
    "first", "then", "therefore", "because", "since",
    "in order to", "goal", "task", "step",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_samples(jsonl_path: str, n_samples: int) -> List[dict]:
    """Load n_samples from the AC training JSONL, filtering for samples with thoughts."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            # Only use samples that have thought content
            if not item.get("has_thought", False):
                continue
            # Need at least one image
            if not item.get("images"):
                continue
            samples.append(item)
            if len(samples) >= n_samples:
                break
    print(f"Loaded {len(samples)} samples with thoughts from {jsonl_path}")
    return samples


def sample_to_qwen_messages(sample: dict) -> Tuple[list, list]:
    """Convert a training sample to Qwen processor message format.

    Returns:
        (qwen_messages, pil_images)
    """
    conversations = sample["conversations"]
    images = sample.get("images", [])

    qwen_messages = []
    pil_images = []
    image_idx = 0

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        content_blocks = []
        # Split on <image> placeholder and interleave image blocks
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if i > 0 and image_idx < len(images):
                # Insert image block before this text part
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

def _match_trigram(input_ids: List[int], i: int, trigram: tuple) -> bool:
    """Check if input_ids[i:i+3] matches a trigram."""
    if i + 2 >= len(input_ids):
        return False
    return (input_ids[i] == trigram[0]
            and input_ids[i + 1] == trigram[1]
            and input_ids[i + 2] == trigram[2])


def find_thought_spans(input_ids: List[int]) -> List[Tuple[int, int]]:
    """Find (start, end) index pairs for <think>...</think> or <thought>...</thought>.

    Uses trigram detection:
      <think>    = (13708, 766, 29)
      </think>   = (522, 26865, 29)
      <thought>  = (13708, 2450, 29)
      </thought> = (522, 60565, 29)
    Returns inclusive spans [start, end].
    """
    spans = []
    i = 0
    n = len(input_ids)
    while i < n - 2:
        if (_match_trigram(input_ids, i, THINK_OPEN_TRIGRAM)
                or _match_trigram(input_ids, i, THOUGHT_OPEN_TRIGRAM)):
            start = i
            # Find matching close
            j = i + 3
            while j < n - 2:
                if (_match_trigram(input_ids, j, THINK_CLOSE_TRIGRAM)
                        or _match_trigram(input_ids, j, THOUGHT_CLOSE_TRIGRAM)):
                    spans.append((start, j + 2))  # inclusive of trigram
                    i = j + 3
                    break
                j += 1
            else:
                # No close found, span extends to end
                spans.append((start, n - 1))
                break
            continue
        i += 1
    return spans


def find_action_spans(input_ids: List[int], tokenizer) -> List[Tuple[int, int]]:
    """Find (start, end) index pairs for <action>...</action> spans.

    Since <action> tokenization varies, we decode the full sequence and use
    regex to find the spans, then map character positions back to token positions.

    Returns inclusive spans [start, end].
    """
    # Decode token by token to build a char-to-token mapping
    # This is simpler: decode the whole thing and use offset tracking
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    spans = []
    for m in re.finditer(r"<action>(.*?)</action>", full_text, re.DOTALL):
        char_start = m.start()
        char_end = m.end() - 1  # inclusive

        # Map char positions to token indices by accumulating decoded lengths
        tok_start = None
        tok_end = None
        cum_len = 0
        for idx, tid in enumerate(input_ids):
            tok_text = tokenizer.decode([tid], skip_special_tokens=False)
            tok_len = len(tok_text)
            tok_char_start = cum_len
            tok_char_end = cum_len + tok_len - 1
            cum_len += tok_len

            if tok_start is None and tok_char_end >= char_start:
                tok_start = idx
            if tok_char_start <= char_end:
                tok_end = idx

        if tok_start is not None and tok_end is not None:
            spans.append((tok_start, tok_end))

    return spans


def find_coord_token_indices(input_ids: List[int]) -> Set[int]:
    """Find indices of coordinate/bbox digit tokens in the sequence.

    Mirrors CooperativeVLMWrapper._mark_coord_tokens logic.
    """
    coord_indices = set()
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
                    coord_indices.add(i)
                else:
                    in_coord = False
                    in_bracket = False

    return coord_indices


def classify_think_token(token_text: str) -> str:
    """Classify a single think token as THINK_SPATIAL or THINK_DECISION.

    Returns "THINK_SPATIAL" or "THINK_DECISION" based on word overlap.
    If ambiguous or empty, defaults to THINK_DECISION.
    """
    text_lower = token_text.lower().strip()
    if not text_lower:
        return "THINK_DECISION"

    spatial_score = sum(1 for w in SPATIAL_WORDS if w in text_lower)
    decision_score = sum(1 for w in DECISION_WORDS if w in text_lower)

    if spatial_score > decision_score:
        return "THINK_SPATIAL"
    return "THINK_DECISION"


def classify_tokens(
    input_ids: List[int],
    tokenizer,
) -> Dict[str, List[int]]:
    """Classify each token index into one of the five types.

    Returns:
        dict mapping type name -> list of token indices
    """
    n = len(input_ids)

    # 1. IMAGE tokens
    image_indices = {i for i in range(n) if input_ids[i] == IMAGE_PAD_ID}

    # 2. COORD tokens (digit tokens inside coordinate/bbox arrays in action)
    coord_indices = find_coord_token_indices(input_ids)
    # Coord tokens should not overlap with image tokens
    coord_indices -= image_indices

    # 3. Thought spans
    thought_spans = find_thought_spans(input_ids)
    thought_indices = set()
    for start, end in thought_spans:
        for i in range(start, end + 1):
            if i not in image_indices and i not in coord_indices:
                thought_indices.add(i)

    # 4. Action spans (non-coord tokens inside <action>...</action>)
    action_spans = find_action_spans(input_ids, tokenizer)
    action_type_indices = set()
    for start, end in action_spans:
        for i in range(start, end + 1):
            if (i not in image_indices
                    and i not in coord_indices
                    and i not in thought_indices):
                action_type_indices.add(i)

    # 5. Sub-classify thought tokens into SPATIAL vs DECISION
    think_spatial = []
    think_decision = []
    # Decode thought tokens in small windows for context-aware classification
    # We use a sliding window of 5 tokens for context
    thought_list = sorted(thought_indices)
    for idx in thought_list:
        # Get a context window around this token
        window_start = max(0, idx - 2)
        window_end = min(n, idx + 3)
        window_ids = input_ids[window_start:window_end]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=False)
        label = classify_think_token(window_text)
        if label == "THINK_SPATIAL":
            think_spatial.append(idx)
        else:
            think_decision.append(idx)

    result = {
        "IMAGE": sorted(image_indices),
        "COORD": sorted(coord_indices),
        "THINK_SPATIAL": think_spatial,
        "THINK_DECISION": think_decision,
        "ACTION_TYPE": sorted(action_type_indices),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════
# Hidden state extraction via forward hooks
# ═══════════════════════════════════════════════════════════════════════

class HiddenStateCollector:
    """Registers forward hooks on specified transformer layers to capture
    hidden states during a single forward pass."""

    def __init__(self, model, layer_indices: List[int]):
        """
        Args:
            model: Qwen2_5_VLForConditionalGeneration
            layer_indices: which layers to hook (0-indexed)
        """
        self.layer_indices = layer_indices
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self._hooks = []

        # Navigate to the transformer layers
        # Qwen2.5-VL structure: model.model.layers (or model.model.language_model.layers)
        vlm = model.model
        if hasattr(vlm, "language_model"):
            layers = vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            layers = vlm.layers
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(vlm).__name__}"
            )

        for li in layer_indices:
            layer = layers[li]
            hook = layer.register_forward_hook(self._make_hook(li))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook function that stores the layer output."""
        def hook_fn(module, input, output):
            # Transformer layer output is typically a tuple; first element
            # is the hidden state tensor of shape [B, seq_len, D].
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            # Detach and move to CPU to save GPU memory
            self.hidden_states[layer_idx] = hs.detach().cpu().float()
        return hook_fn

    def clear(self):
        """Clear stored hidden states for next sample."""
        self.hidden_states.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════
# Analysis utilities
# ═══════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_pairwise_cosine(type_means: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute pairwise cosine similarities between all type mean vectors."""
    types = sorted(type_means.keys())
    results = {}
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            key = f"{types[i]} vs {types[j]}"
            results[key] = cosine_sim(type_means[types[i]], type_means[types[j]])
    return results


def linear_probe_where_vs_what(
    hidden_vecs: Dict[str, np.ndarray],
) -> Tuple[float, int]:
    """Train a simple logistic regression to classify WHERE vs WHAT.

    WHERE = IMAGE + COORD + THINK_SPATIAL  (label 1)
    WHAT  = THINK_DECISION + ACTION_TYPE   (label 0)

    Uses sklearn if available, otherwise falls back to a simple
    closed-form least-squares classifier.

    Returns:
        (accuracy, n_samples)
    """
    where_types = {"IMAGE", "COORD", "THINK_SPATIAL"}
    what_types = {"THINK_DECISION", "ACTION_TYPE"}

    X_list = []
    y_list = []

    for ttype, vecs in hidden_vecs.items():
        if vecs.shape[0] == 0:
            continue
        if ttype in where_types:
            X_list.append(vecs)
            y_list.append(np.ones(vecs.shape[0], dtype=np.int64))
        elif ttype in what_types:
            X_list.append(vecs)
            y_list.append(np.zeros(vecs.shape[0], dtype=np.int64))

    if not X_list:
        return 0.0, 0

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if len(np.unique(y)) < 2:
        return 0.0, len(y)

    # Standardize features for numerical stability
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Use 5-fold CV if enough samples, otherwise leave-one-out
        n = len(y)
        cv = min(5, n) if n >= 10 else max(2, n)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")
        return float(scores.mean()), n

    except ImportError:
        # Fallback: least-squares classifier (no CV, train=test)
        # Add bias column
        X_b = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])
        # Solve X_b @ w = y via pseudoinverse
        w, _, _, _ = np.linalg.lstsq(X_b, y.astype(np.float64), rcond=None)
        preds = (X_b @ w > 0.5).astype(np.int64)
        acc = (preds == y).mean()
        return float(acc), len(y)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layer_indices = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("Hidden State Routing Verification")
    print("=" * 70)
    print(f"Base model:  {args.base_model}")
    print(f"Data:        {args.data}")
    print(f"N samples:   {args.n_samples}")
    print(f"Layers:      {layer_indices}")
    print(f"Device:      {device}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("\nLoading base model (no LoRA)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    # ── Load data ─────────────────────────────────────────────────────
    samples = load_samples(args.data, args.n_samples)
    if not samples:
        print("ERROR: No valid samples found. Exiting.")
        return

    # ── Register hooks ────────────────────────────────────────────────
    collector = HiddenStateCollector(model, layer_indices)

    # ── Accumulate hidden states per type per layer ───────────────────
    # Structure: layer_idx -> type_name -> list of np.ndarray vectors
    all_vectors: Dict[int, Dict[str, List[np.ndarray]]] = {
        li: defaultdict(list) for li in layer_indices
    }

    # Token count statistics
    token_counts: Dict[str, int] = defaultdict(int)

    print(f"\nProcessing {len(samples)} samples...")
    for si, sample in enumerate(samples):
        # Convert sample to Qwen messages
        qwen_messages, pil_images = sample_to_qwen_messages(sample)

        # Apply chat template to get the full text
        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=False,
        )

        # Process through the processor (handles image resizing, tokenization)
        if pil_images:
            inputs = processor(
                text=[text], images=pil_images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = processor(
                text=[text],
                return_tensors="pt", padding=True,
            )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"][0].tolist()  # [seq_len]
        seq_len = len(input_ids)

        # Classify tokens
        token_types = classify_tokens(input_ids, tokenizer)

        # Count tokens
        for ttype, indices in token_types.items():
            token_counts[ttype] += len(indices)

        # Check we have tokens of interest
        total_classified = sum(len(v) for v in token_types.values())
        if total_classified == 0:
            print(f"  Sample {si}: no classifiable tokens, skipping")
            continue

        # Forward pass to collect hidden states
        collector.clear()
        with torch.no_grad():
            _ = model(**inputs)

        # Extract and store per-type vectors for each layer
        for li in layer_indices:
            if li not in collector.hidden_states:
                print(f"  WARNING: layer {li} not captured for sample {si}")
                continue
            hs = collector.hidden_states[li][0]  # [seq_len, D], already float32 on CPU

            for ttype, indices in token_types.items():
                if not indices:
                    continue
                # Clamp indices to valid range (safety check)
                valid_indices = [i for i in indices if i < hs.shape[0]]
                if not valid_indices:
                    continue
                vecs = hs[valid_indices].numpy()  # [n_tokens, D]
                all_vectors[li][ttype].append(vecs)

        # Progress
        type_summary = ", ".join(
            f"{k}={len(v)}" for k, v in token_types.items() if v
        )
        print(f"  Sample {si}: seq_len={seq_len}, {type_summary}")

    # Clean up hooks
    collector.remove_hooks()

    # ═══════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("TOKEN COUNT SUMMARY")
    print("=" * 70)
    for ttype in ["IMAGE", "COORD", "THINK_SPATIAL", "THINK_DECISION", "ACTION_TYPE"]:
        count = token_counts.get(ttype, 0)
        group = "WHERE" if ttype in {"IMAGE", "COORD", "THINK_SPATIAL"} else "WHAT"
        print(f"  {ttype:20s}: {count:6d} tokens  ({group})")
    total_where = sum(token_counts.get(t, 0) for t in ["IMAGE", "COORD", "THINK_SPATIAL"])
    total_what = sum(token_counts.get(t, 0) for t in ["THINK_DECISION", "ACTION_TYPE"])
    print(f"  {'TOTAL WHERE':20s}: {total_where:6d}")
    print(f"  {'TOTAL WHAT':20s}: {total_what:6d}")

    print("\n" + "=" * 70)
    print("PER-LAYER ANALYSIS")
    print("=" * 70)

    for li in layer_indices:
        print(f"\n{'─' * 60}")
        print(f"LAYER {li}")
        print(f"{'─' * 60}")

        # Concatenate all vectors per type
        type_vecs: Dict[str, np.ndarray] = {}
        type_means: Dict[str, np.ndarray] = {}

        for ttype in ["IMAGE", "COORD", "THINK_SPATIAL", "THINK_DECISION", "ACTION_TYPE"]:
            vec_list = all_vectors[li].get(ttype, [])
            if vec_list:
                cat = np.concatenate(vec_list, axis=0)
                type_vecs[ttype] = cat
                type_means[ttype] = cat.mean(axis=0)
                print(f"  {ttype:20s}: {cat.shape[0]:5d} vectors, "
                      f"mean norm = {np.linalg.norm(type_means[ttype]):.4f}")
            else:
                print(f"  {ttype:20s}: no vectors")

        if len(type_means) < 2:
            print("  Not enough types with data for comparison.")
            continue

        # ── Pairwise cosine similarity ────────────────────────────────
        print(f"\n  Pairwise cosine similarity:")
        cos_sims = compute_pairwise_cosine(type_means)
        for pair, sim in sorted(cos_sims.items()):
            # Highlight WHERE-vs-WHAT pairs
            parts = pair.split(" vs ")
            where_types = {"IMAGE", "COORD", "THINK_SPATIAL"}
            what_types = {"THINK_DECISION", "ACTION_TYPE"}
            is_cross = ((parts[0] in where_types and parts[1] in what_types)
                        or (parts[0] in what_types and parts[1] in where_types))
            marker = " [WHERE-WHAT]" if is_cross else ""
            print(f"    {pair:40s}: {sim:+.4f}{marker}")

        # ── WHERE vs WHAT group means ─────────────────────────────────
        where_vecs = []
        what_vecs = []
        for ttype, vecs in type_vecs.items():
            if ttype in {"IMAGE", "COORD", "THINK_SPATIAL"}:
                where_vecs.append(vecs)
            elif ttype in {"THINK_DECISION", "ACTION_TYPE"}:
                what_vecs.append(vecs)

        if where_vecs and what_vecs:
            where_all = np.concatenate(where_vecs, axis=0)
            what_all = np.concatenate(what_vecs, axis=0)
            where_mean = where_all.mean(axis=0)
            what_mean = what_all.mean(axis=0)
            group_cos = cosine_sim(where_mean, what_mean)
            print(f"\n  GROUP cosine (WHERE mean vs WHAT mean): {group_cos:+.4f}")

            # Intra-group similarity (how tight is each cluster?)
            # Sample up to 500 vectors for speed
            n_sample = 500
            if where_all.shape[0] > n_sample:
                idx = np.random.choice(where_all.shape[0], n_sample, replace=False)
                where_sample = where_all[idx]
            else:
                where_sample = where_all
            if what_all.shape[0] > n_sample:
                idx = np.random.choice(what_all.shape[0], n_sample, replace=False)
                what_sample = what_all[idx]
            else:
                what_sample = what_all

            # Intra-group: average pairwise cosine within each group (sample-based)
            n_pairs = min(200, where_sample.shape[0] * (where_sample.shape[0] - 1) // 2)
            if where_sample.shape[0] >= 2 and n_pairs > 0:
                # Compute mean cosine within WHERE
                norms = np.linalg.norm(where_sample, axis=1, keepdims=True) + 1e-10
                where_normed = where_sample / norms
                # Random pairs
                rng = np.random.default_rng(42)
                i1 = rng.integers(0, where_sample.shape[0], size=n_pairs)
                i2 = rng.integers(0, where_sample.shape[0], size=n_pairs)
                mask = i1 != i2
                i1, i2 = i1[mask], i2[mask]
                if len(i1) > 0:
                    intra_where = (where_normed[i1] * where_normed[i2]).sum(axis=1).mean()
                    print(f"  Intra-WHERE avg cosine (sampled):       {intra_where:+.4f}")

            if what_sample.shape[0] >= 2:
                norms = np.linalg.norm(what_sample, axis=1, keepdims=True) + 1e-10
                what_normed = what_sample / norms
                rng = np.random.default_rng(42)
                i1 = rng.integers(0, what_sample.shape[0], size=n_pairs)
                i2 = rng.integers(0, what_sample.shape[0], size=n_pairs)
                mask = i1 != i2
                i1, i2 = i1[mask], i2[mask]
                if len(i1) > 0:
                    intra_what = (what_normed[i1] * what_normed[i2]).sum(axis=1).mean()
                    print(f"  Intra-WHAT avg cosine (sampled):        {intra_what:+.4f}")

        # ── Linear probe ──────────────────────────────────────────────
        # Subsample to keep probe fast (max 2000 vectors per type)
        max_per_type = 2000
        probe_vecs: Dict[str, np.ndarray] = {}
        for ttype, vecs in type_vecs.items():
            if vecs.shape[0] > max_per_type:
                idx = np.random.choice(vecs.shape[0], max_per_type, replace=False)
                probe_vecs[ttype] = vecs[idx]
            else:
                probe_vecs[ttype] = vecs

        acc, n_probe = linear_probe_where_vs_what(probe_vecs)
        print(f"\n  Linear probe (WHERE vs WHAT): accuracy = {acc:.4f}  "
              f"(n={n_probe})")

    # ═══════════════════════════════════════════════════════════════════
    # Final summary table
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("SUMMARY: WHERE vs WHAT Separation Across Layers")
    print("=" * 70)
    header = f"{'Layer':>6s} | {'Group CosSim':>12s} | {'Probe Acc':>10s} | {'N probe':>8s}"
    print(header)
    print("-" * len(header))

    for li in layer_indices:
        # Recompute group cosine and probe accuracy for summary
        type_vecs_summary: Dict[str, np.ndarray] = {}
        for ttype in ["IMAGE", "COORD", "THINK_SPATIAL", "THINK_DECISION", "ACTION_TYPE"]:
            vec_list = all_vectors[li].get(ttype, [])
            if vec_list:
                type_vecs_summary[ttype] = np.concatenate(vec_list, axis=0)

        where_vecs = [v for t, v in type_vecs_summary.items()
                      if t in {"IMAGE", "COORD", "THINK_SPATIAL"}]
        what_vecs = [v for t, v in type_vecs_summary.items()
                     if t in {"THINK_DECISION", "ACTION_TYPE"}]

        if where_vecs and what_vecs:
            where_mean = np.concatenate(where_vecs, axis=0).mean(axis=0)
            what_mean = np.concatenate(what_vecs, axis=0).mean(axis=0)
            group_cos = cosine_sim(where_mean, what_mean)
        else:
            group_cos = float("nan")

        # Probe
        max_per_type = 2000
        probe_vecs_s: Dict[str, np.ndarray] = {}
        for ttype, vecs in type_vecs_summary.items():
            if vecs.shape[0] > max_per_type:
                idx = np.random.choice(vecs.shape[0], max_per_type, replace=False)
                probe_vecs_s[ttype] = vecs[idx]
            else:
                probe_vecs_s[ttype] = vecs
        acc, n_probe = linear_probe_where_vs_what(probe_vecs_s)

        print(f"{li:>6d} | {group_cos:>+12.4f} | {acc:>10.4f} | {n_probe:>8d}")

    print("\n" + "=" * 70)
    print("Interpretation guide:")
    print("  - Group CosSim close to 1.0 => WHERE and WHAT are similar (bad for routing)")
    print("  - Group CosSim << 1.0       => WHERE and WHAT are already separated (good)")
    print("  - Probe Acc >> 0.5          => linear separability exists (good for routing)")
    print("  - Probe Acc ~ 0.5           => no linear signal (routing must learn it)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify hidden state WHERE/WHAT separation in base Qwen2.5-VL"
    )
    parser.add_argument(
        "--base_model", type=str,
        default="checkpoints/Qwen2.5-VL-7B-Instruct",
        help="Path to base Qwen2.5-VL model",
    )
    parser.add_argument(
        "--data", type=str,
        default="datasets/cooperative_thought_ac/ac_train_thought.jsonl",
        help="Path to training JSONL with thought annotations",
    )
    parser.add_argument(
        "--n_samples", type=int, default=15,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--layers", type=str, default="0,7,14,21,27",
        help="Comma-separated layer indices to probe",
    )
    args = parser.parse_args()
    main(args)
