# Amendment 005: Selector Implementation Details

Date: 2026-08-12

Timing: before public-bank creation, private-label creation, or selector inference.

The public-bank fold map is frozen to `runs/complementarity/2026-07-30/folds.json`, SHA-256 `0637c9d91bb2ba676802f98f9313fcfd8e9da04e971e90ab52217a008429fbec`. Low and High paired records must have identical row IDs, image hashes, episode groups, and folds.

Candidate normalization is source-agnostic:

- preserve the parsed canonical action string; missing action becomes `UNKNOWN`;
- expose normalized coordinate only for click, long_press, moveto, doubleclick, or rightclick;
- expose parameter only for type, open_app, scroll, or select;
- all other coordinates/parameters become null/empty;
- clip candidate parameters to 256 characters and compact history to the final 512 characters;
- preserve parse-ok;
- never expose raw response, source/model/slot identity, model prompt, reliability, or provenance hashes as model inputs.

Before public serialization, the three private canonical candidates are reordered by a SHA-256 permutation of `(sample_key, seed, public-bank-order)`. The private source mapping is recoverable later from the frozen seed but is absent from the public file. Every VLM query then applies a second independent `(sample_key, seed, selector-display)` permutation. Public order therefore cannot act as a source-position side channel.

The selector interpreter is pinned to `.venv-scaleup/bin/python`, and the eight-shard GPU map must be an exact bijection over physical GPUs 0--7. Normalized coordinates outside `[0,1]` are rejected; a result-blind audit found zero such values among 12,000 recovered candidates.

Each prediction stores a hash of the exact prompt and rendered overlay RGB content. The blind finalizer reconstructs both from the public record and display permutation, verifies both hashes, and independently recomputes the selected-label probability argmax.