import os
import hashlib
from huggingface_hub import snapshot_download

# Download ShowUI-2B
print("Downloading ShowUI-2B...")
showui_dir = snapshot_download(
    repo_id="showlab/ShowUI-2B",
    revision="cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60",
    local_dir="models/ShowUI-2B",
    local_dir_use_symlinks=False
)
print(f"ShowUI-2B downloaded to {showui_dir}")

# Download Qwen2-VL-2B-Instruct-processor
print("Downloading Qwen2-VL-2B-Instruct-processor...")
qwen_dir = snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    revision="895c3a49bc3fa70a340399125c650a463535e71c",
    local_dir="models/Qwen2-VL-2B-Instruct-processor",
    allow_patterns=[
        "chat_template.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json", "config.json",
        "generation_config.json"
    ],
    local_dir_use_symlinks=False
)
print(f"Qwen2-VL-2B-Instruct-processor downloaded to {qwen_dir}")
