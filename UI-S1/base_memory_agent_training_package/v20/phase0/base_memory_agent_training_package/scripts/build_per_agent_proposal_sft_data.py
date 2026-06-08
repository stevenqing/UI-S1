#!/usr/bin/env python3
"""Package wrapper for per-agent proposal SFT data generation.

Run this from the repository root. It delegates to the canonical implementation
at v20/phase0/memory_controller/build_per_agent_proposal_sft_data.py so the
training package keeps a stable entry point while avoiding duplicate logic.
"""
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "memory_controller" / "build_per_agent_proposal_sft_data.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
