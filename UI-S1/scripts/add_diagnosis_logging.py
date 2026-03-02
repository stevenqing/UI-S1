#!/usr/bin/env python3
"""
Script to add diagnosis logging to FSDP sharding managers.
This helps identify where multi-node training hangs.

Usage:
    python scripts/add_diagnosis_logging.py --enable   # Add logging
    python scripts/add_diagnosis_logging.py --disable  # Remove logging
"""

import argparse
import re
import shutil
from pathlib import Path

VLLM_FILE = Path("verl/workers/sharding_manager/fsdp_vllm.py")
SGLANG_FILE = Path("verl/workers/sharding_manager/fsdp_sglang.py")

# Diagnosis code to insert
DIAGNOSIS_IMPORTS = '''
# === DIAGNOSIS IMPORTS (auto-added) ===
import time as _diag_time
import torch.distributed as _diag_dist
def _diag_log(msg):
    rank = _diag_dist.get_rank() if _diag_dist.is_initialized() else 0
    print(f"[DIAG][Rank {rank}][{_diag_time.time():.3f}] {msg}", flush=True)
# === END DIAGNOSIS IMPORTS ===
'''

DIAGNOSIS_MARKER_START = "# === DIAGNOSIS"
DIAGNOSIS_MARKER_END = "# === END DIAGNOSIS"


def add_diagnosis_to_vllm():
    """Add diagnosis logging to fsdp_vllm.py"""
    if not VLLM_FILE.exists():
        print(f"File not found: {VLLM_FILE}")
        return False

    # Backup
    backup_file = VLLM_FILE.with_suffix(".py.bak")
    shutil.copy(VLLM_FILE, backup_file)
    print(f"Backup created: {backup_file}")

    content = VLLM_FILE.read_text()

    # Check if already added
    if DIAGNOSIS_MARKER_START in content:
        print(f"Diagnosis logging already present in {VLLM_FILE}")
        return True

    # Add imports after the existing imports
    import_insert_pos = content.find("from .base import BaseShardingManager")
    if import_insert_pos == -1:
        print("Could not find import insertion point")
        return False

    import_insert_pos = content.find("\n", import_insert_pos) + 1
    content = content[:import_insert_pos] + DIAGNOSIS_IMPORTS + content[import_insert_pos:]

    # Add logging in __enter__ method
    # Find the __enter__ method and add logging at key points

    # 1. After "self.timing = {}"
    content = content.replace(
        'self.timing = {}',
        '''self.timing = {}
        _diag_log("__enter__ started")'''
    )

    # 2. Before load_fsdp_model_to_gpu
    content = content.replace(
        'if self.offload_param:\n                load_fsdp_model_to_gpu(self.module)',
        '''_diag_log("Before offload check")
            if self.offload_param:
                _diag_log("Starting load_fsdp_model_to_gpu")
                load_fsdp_model_to_gpu(self.module)
                _diag_log("Finished load_fsdp_model_to_gpu")
                # Sync after loading to GPU
                if _diag_dist.is_initialized():
                    _diag_log("Waiting at barrier after load_fsdp_model_to_gpu")
                    _diag_dist.barrier()
                    _diag_log("Passed barrier after load_fsdp_model_to_gpu")'''
    )

    # 3. Before state_dict() call (non-peft path)
    content = content.replace(
        'else:\n                params = self.module.state_dict()',
        '''else:
                _diag_log("Before state_dict() call")
                if _diag_dist.is_initialized():
                    _diag_log("Waiting at barrier before state_dict")
                    _diag_dist.barrier()
                    _diag_log("Passed barrier, now calling state_dict")
                params = self.module.state_dict()
                _diag_log("After state_dict() call")'''
    )

    # 4. After convert_weight_keys
    content = content.replace(
        'params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))',
        '''params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
            _diag_log(f"convert_weight_keys done, params count: {len(params)}")'''
    )

    VLLM_FILE.write_text(content)
    print(f"Diagnosis logging added to {VLLM_FILE}")
    return True


def add_diagnosis_to_sglang():
    """Add diagnosis logging to fsdp_sglang.py"""
    if not SGLANG_FILE.exists():
        print(f"File not found: {SGLANG_FILE}")
        return False

    # Backup
    backup_file = SGLANG_FILE.with_suffix(".py.bak")
    shutil.copy(SGLANG_FILE, backup_file)
    print(f"Backup created: {backup_file}")

    content = SGLANG_FILE.read_text()

    # Check if already added
    if DIAGNOSIS_MARKER_START in content:
        print(f"Diagnosis logging already present in {SGLANG_FILE}")
        return True

    # Add imports after the existing imports
    import_insert_pos = content.find("from .base import BaseShardingManager")
    if import_insert_pos == -1:
        print("Could not find import insertion point")
        return False

    import_insert_pos = content.find("\n", import_insert_pos) + 1
    content = content[:import_insert_pos] + DIAGNOSIS_IMPORTS + content[import_insert_pos:]

    # Add logging in __enter__ method
    content = content.replace(
        'self.timing = {}',
        '''self.timing = {}
        _diag_log("__enter__ started")'''
    )

    content = content.replace(
        'if self.offload_param:\n                load_fsdp_model_to_gpu(self.module)',
        '''_diag_log("Before offload check")
            if self.offload_param:
                _diag_log("Starting load_fsdp_model_to_gpu")
                load_fsdp_model_to_gpu(self.module)
                _diag_log("Finished load_fsdp_model_to_gpu")
                if _diag_dist.is_initialized():
                    _diag_log("Waiting at barrier after load_fsdp_model_to_gpu")
                    _diag_dist.barrier()
                    _diag_log("Passed barrier after load_fsdp_model_to_gpu")'''
    )

    content = content.replace(
        'params = self.module.state_dict()',
        '''_diag_log("Before state_dict() call")
            if _diag_dist.is_initialized():
                _diag_log("Waiting at barrier before state_dict")
                _diag_dist.barrier()
                _diag_log("Passed barrier, now calling state_dict")
            params = self.module.state_dict()
            _diag_log("After state_dict() call")''',
        1  # Only replace first occurrence
    )

    SGLANG_FILE.write_text(content)
    print(f"Diagnosis logging added to {SGLANG_FILE}")
    return True


def remove_diagnosis(filepath: Path):
    """Remove diagnosis logging from a file"""
    if not filepath.exists():
        return

    content = filepath.read_text()

    if DIAGNOSIS_MARKER_START not in content:
        print(f"No diagnosis logging found in {filepath}")
        return

    # Remove diagnosis imports block
    pattern = rf'{re.escape(DIAGNOSIS_MARKER_START)}.*?{re.escape(DIAGNOSIS_MARKER_END)}[^\n]*\n'
    content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Remove _diag_log calls
    content = re.sub(r'\s*_diag_log\([^)]+\)\n?', '\n', content)

    # Remove barrier blocks added for diagnosis
    content = re.sub(
        r'\s*if _diag_dist\.is_initialized\(\):\s*\n\s*_diag_dist\.barrier\(\)\s*\n?',
        '\n',
        content
    )

    # Restore original structure (this is approximate)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

    filepath.write_text(content)
    print(f"Diagnosis logging removed from {filepath}")

    # Check for backup
    backup_file = filepath.with_suffix(".py.bak")
    if backup_file.exists():
        print(f"Note: Backup file exists at {backup_file}")
        print(f"To fully restore: cp {backup_file} {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Add/remove diagnosis logging for multi-node hang debugging")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true", help="Add diagnosis logging")
    group.add_argument("--disable", action="store_true", help="Remove diagnosis logging")
    parser.add_argument("--file", choices=["vllm", "sglang", "both"], default="both",
                       help="Which sharding manager to modify (default: both)")

    args = parser.parse_args()

    if args.enable:
        print("Adding diagnosis logging...")
        if args.file in ["vllm", "both"]:
            add_diagnosis_to_vllm()
        if args.file in ["sglang", "both"]:
            add_diagnosis_to_sglang()
        print("\nDiagnosis logging enabled. Run your training and check for [DIAG] prefixed logs.")
        print("The logs will show timestamps for each critical operation to identify where hangs occur.")

    elif args.disable:
        print("Removing diagnosis logging...")
        if args.file in ["vllm", "both"]:
            remove_diagnosis(VLLM_FILE)
        if args.file in ["sglang", "both"]:
            remove_diagnosis(SGLANG_FILE)
        print("\nDiagnosis logging disabled.")
        print("Consider restoring from backup files (.py.bak) for a clean state.")


if __name__ == "__main__":
    main()
