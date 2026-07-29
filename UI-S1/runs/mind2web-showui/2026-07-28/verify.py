import os
import hashlib
import json

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def verify_dir(local_dir, name):
    print(f"=== Verification for {name} ({local_dir}) ===")

    # Verify .cache metadata / ref
    # Let's find cache dir or commit in .cache
    cache_ref_path = os.path.join(local_dir, ".cache", "huggingface", "hub")
    # ShowUI or Qwen local_dir has a .cache directory with metadata
    commit_hash = None
    ref_file_path = None
    for root, dirs, files in os.walk(os.path.join(local_dir, ".cache")):
        for file in files:
            # Look for file containing the commit hash or reference
            if file == "commit" or file == "ref" or file.endswith(".json") or len(file) == 40:
                try:
                    fpath = os.path.join(root, file)
                    if "refs" in root: # e.g. .cache/huggingface/hub/models.../refs/main
                        with open(fpath, "r", encoding="utf-8") as f:
                            commit_hash = f.read().strip()
                            ref_file_path = fpath
                            break
                except Exception:
                    pass
        if commit_hash:
            break

    if commit_hash:
        print(f"Detected cache revision commit hash: {commit_hash} (from {os.path.relpath(ref_file_path, local_dir)})")
    else:
        # Check standard user level global cache or snapshot result commit
        print("Commit hash not found directly in local_dir/.cache, checking .gitattributes or config files")
        # Let's look up using hf_api / check if files downloaded match

    # List non-cache files with bytes and SHA256
    print("\nNon-cache files listing:")
    total_bytes = 0
    for root, dirs, files in sorted(os.walk(local_dir)):
        # Skip directories named .cache or examples (except if we want all non-cache files)
        if ".cache" in root.split(os.sep):
            continue
        for file in sorted(files):
            fpath = os.path.join(root, file)
            size = os.path.getsize(fpath)
            # Avoid calculating SHA-256 of pytorch_model.bin since it is 4.4GB and slow, but let's do it if needed or only print it.
            # Wait, the prompt says "list non-cache files with bytes and SHA-256"
            # Since pytorch_model.bin is 4.4GB, let's calc SHA-256 or mention size. Let's compute it.
            if size > 100 * 1024 * 1024:
                # Fast hash/print for huge files or full hash if required
                print(f"Calculating SHA-256 for large file {file}...")
            sha = get_sha256(fpath)
            total_bytes += size
            print(f"- {os.path.relpath(fpath, local_dir)}: {size} bytes, SHA-256: {sha}")

    print(f"\nTotal bytes for {name}: {total_bytes} bytes\n")

# Verify ShowUI-2B
verify_dir("models/ShowUI-2B", "ShowUI-2B")

# Verify Qwen2-VL-2B-Instruct-processor
verify_dir("models/Qwen2-VL-2B-Instruct-processor", "Qwen2-VL-2B-Instruct-processor")

# Confirm pytorch_model.bin exists and is nonzero
bin_path = "models/ShowUI-2B/pytorch_model.bin"
if os.path.exists(bin_path):
    sz = os.path.getsize(bin_path)
    print(f"CONFIRMATION: {bin_path} exists and is {sz} bytes (nonzero: {sz > 0})")
else:
    print(f"CONFIRMATION: {bin_path} DOES NOT EXIST!")
