"""V22: Memory retrieval functions.

4 retrieval strategies (Dimension 1):
  - Goal: TF-IDF cosine similarity on instruction text
  - Procedural: Jaccard similarity on action-type bigram sequences
  - Type: Filter by (app, action_type) match
  - Visual: App + normalized step-position proximity + goal similarity

Each returns K=3 examples per query step, adding ~250 tokens to the prompt.
"""

import json
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryIndex:
    """Loads and caches memory indices for retrieval."""

    def __init__(self, index_dir: str = "v22_memory_agent/indices"):
        self.index_dir = index_dir
        self._step_records = None
        self._episode_metadata = None
        self._vectorizer = None
        self._tfidf_matrix = None
        self._type_index = None

    @property
    def step_records(self) -> list:
        if self._step_records is None:
            path = os.path.join(self.index_dir, "step_records.json")
            with open(path, "r") as f:
                self._step_records = json.load(f)
        return self._step_records

    @property
    def episode_metadata(self) -> list:
        if self._episode_metadata is None:
            path = os.path.join(self.index_dir, "episode_metadata.json")
            with open(path, "r") as f:
                self._episode_metadata = json.load(f)
        return self._episode_metadata

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            path = os.path.join(self.index_dir, "goal_vectorizer.pkl")
            with open(path, "rb") as f:
                self._vectorizer = pickle.load(f)
        return self._vectorizer

    @property
    def tfidf_matrix(self):
        if self._tfidf_matrix is None:
            path = os.path.join(self.index_dir, "goal_tfidf_matrix.pkl")
            with open(path, "rb") as f:
                self._tfidf_matrix = pickle.load(f)
        return self._tfidf_matrix

    @property
    def type_index(self) -> dict:
        if self._type_index is None:
            path = os.path.join(self.index_dir, "type_index.json")
            with open(path, "r") as f:
                self._type_index = json.load(f)
        return self._type_index


def retrieve_goal(
    index: MemoryIndex,
    query_goal: str,
    query_step_idx: int,
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
) -> List[Dict]:
    """Retrieve K most similar steps by goal TF-IDF cosine similarity.

    Matches at the episode level, then picks the step closest to query_step_idx
    from each matched episode.
    """
    # Transform query goal
    query_vec = index.vectorizer.transform([query_goal])
    sims = cosine_similarity(query_vec, index.tfidf_matrix).flatten()

    # Rank episodes by similarity
    ranked_ep_indices = np.argsort(-sims)

    results = []
    for ep_idx in ranked_ep_indices:
        if len(results) >= k:
            break
        ep_meta = index.episode_metadata[ep_idx]
        if exclude_episode_id is not None and ep_meta["episode_id"] == exclude_episode_id:
            continue

        sim_score = float(sims[ep_idx])
        if sim_score <= 0:
            break

        # Find the step in this episode closest to query_step_idx
        best_record = _find_closest_step(index, ep_meta["episode_id"], query_step_idx)
        if best_record:
            best_record["similarity"] = sim_score
            results.append(best_record)

    return results


def retrieve_procedural(
    index: MemoryIndex,
    query_action_sequence: List[str],
    query_step_idx: int,
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
) -> List[Dict]:
    """Retrieve K most similar steps by Jaccard similarity on action-type bigrams.

    Compares the action-type bigram set of the query episode's history
    with each training episode's full action-type sequence.
    """
    query_bigrams = _get_bigrams(query_action_sequence)
    if not query_bigrams:
        # Fallback: use unigrams if sequence is too short
        query_set = set(query_action_sequence)
    else:
        query_set = query_bigrams

    scored = []
    for ep_idx, ep_meta in enumerate(index.episode_metadata):
        if exclude_episode_id is not None and ep_meta["episode_id"] == exclude_episode_id:
            continue
        ep_seq = ep_meta["action_type_sequence"]
        if not ep_seq:
            continue
        ep_bigrams = _get_bigrams(ep_seq) if query_bigrams else set(ep_seq)
        if not ep_bigrams:
            continue
        jaccard = len(query_set & ep_bigrams) / len(query_set | ep_bigrams)
        if jaccard > 0:
            scored.append((ep_idx, jaccard))

    scored.sort(key=lambda x: -x[1])

    results = []
    for ep_idx, sim_score in scored[:k * 2]:  # over-fetch to handle filtering
        if len(results) >= k:
            break
        ep_meta = index.episode_metadata[ep_idx]
        best_record = _find_closest_step(index, ep_meta["episode_id"], query_step_idx)
        if best_record:
            best_record["similarity"] = sim_score
            results.append(best_record)

    return results


def retrieve_type(
    index: MemoryIndex,
    query_app: str,
    query_action_type: str,
    query_step_idx: int,
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
) -> List[Dict]:
    """Retrieve K steps matching (app, action_type), preferring similar step position.

    Falls back to action_type-only match if exact (app, type) has too few results.
    """
    type_key = f"{query_app}|{query_action_type}"
    candidates = index.type_index.get(type_key, [])

    # Fallback: try any app with this action type
    if len(candidates) < k:
        for key, indices in index.type_index.items():
            if key.endswith(f"|{query_action_type}") and key != type_key:
                candidates.extend(indices)

    # Score by step position proximity
    scored = []
    for rec_idx in candidates:
        rec = index.step_records[rec_idx]
        if exclude_episode_id is not None and rec["episode_id"] == exclude_episode_id:
            continue
        # Prefer similar step position (normalized)
        pos_sim = 1.0 / (1.0 + abs(rec["step_idx"] - query_step_idx))
        scored.append((rec_idx, pos_sim))

    scored.sort(key=lambda x: -x[1])

    # Deduplicate by episode_id
    seen_episodes = set()
    results = []
    for rec_idx, sim_score in scored:
        if len(results) >= k:
            break
        rec = index.step_records[rec_idx]
        if rec["episode_id"] in seen_episodes:
            continue
        seen_episodes.add(rec["episode_id"])
        result = dict(rec)
        result["similarity"] = sim_score
        results.append(result)

    return results


def retrieve_visual(
    index: MemoryIndex,
    query_goal: str,
    query_app: str,
    query_step_idx: int,
    query_num_steps: int,
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
) -> List[Dict]:
    """Retrieve K steps by app match + normalized step-position proximity + goal similarity.

    Combined score = 0.5 * goal_cosine + 0.3 * app_match + 0.2 * position_proximity
    """
    # Goal similarity
    query_vec = index.vectorizer.transform([query_goal])
    goal_sims = cosine_similarity(query_vec, index.tfidf_matrix).flatten()

    # Normalized step position
    query_pos = query_step_idx / max(query_num_steps, 1)

    scored = []
    for ep_idx, ep_meta in enumerate(index.episode_metadata):
        if exclude_episode_id is not None and ep_meta["episode_id"] == exclude_episode_id:
            continue

        goal_sim = float(goal_sims[ep_idx])
        app_match = 1.0 if ep_meta["app"] == query_app else 0.0
        ep_num_steps = ep_meta["num_steps"]
        ep_pos = query_step_idx / max(ep_num_steps, 1)  # same absolute step in this episode
        pos_proximity = 1.0 / (1.0 + abs(query_pos - ep_pos) * 5)

        combined = 0.5 * goal_sim + 0.3 * app_match + 0.2 * pos_proximity
        if combined > 0.05:
            scored.append((ep_idx, combined))

    scored.sort(key=lambda x: -x[1])

    results = []
    for ep_idx, sim_score in scored[:k * 2]:
        if len(results) >= k:
            break
        ep_meta = index.episode_metadata[ep_idx]
        best_record = _find_closest_step(index, ep_meta["episode_id"], query_step_idx)
        if best_record:
            best_record["similarity"] = sim_score
            results.append(best_record)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_closest_step(index: MemoryIndex, episode_id: int, target_step_idx: int) -> Optional[Dict]:
    """Find the step record in a given episode closest to target_step_idx."""
    best = None
    best_dist = float("inf")
    for rec in index.step_records:
        if rec["episode_id"] == episode_id:
            dist = abs(rec["step_idx"] - target_step_idx)
            if dist < best_dist:
                best_dist = dist
                best = rec
    if best is None:
        return None
    return dict(best)  # copy


def _get_bigrams(sequence: List[str]) -> set:
    """Get bigram set from a sequence of action types."""
    if len(sequence) < 2:
        return set()
    return {(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)}


def retrieve_random(
    index: MemoryIndex,
    query_step_idx: int,
    k: int = 3,
    exclude_episode_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Retrieve K random steps (ablation control).

    Returns random training examples with similarity=0.5 (fixed).
    Uses deterministic seed per (episode_id, step_idx) for reproducibility.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Collect candidate indices excluding the query episode
    candidates = list(range(len(index.step_records)))
    if exclude_episode_id is not None:
        candidates = [i for i in candidates
                      if index.step_records[i]["episode_id"] != exclude_episode_id]

    chosen = rng.sample(candidates, min(k, len(candidates)))

    results = []
    for rec_idx in chosen:
        rec = dict(index.step_records[rec_idx])
        rec["similarity"] = 0.5  # fixed weight
        results.append(rec)

    return results


# Convenience: map memory type name -> retrieval function signature
MEMORY_TYPES = ["goal", "procedural", "type", "visual", "random"]
