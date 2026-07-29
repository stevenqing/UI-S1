import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


BUCKET = "gresearch"
PREFIX = "android_control/"
LIST_URL = (
    "https://storage.googleapis.com/storage/v1/b/"
    f"{BUCKET}/o?prefix={urllib.parse.quote(PREFIX, safe='')}"
)


def fetch_objects() -> list[dict]:
    with urllib.request.urlopen(LIST_URL) as response:
        payload = json.load(response)
    objects = []
    for item in payload["items"]:
        name = item["name"]
        if name.startswith(f"{PREFIX}android_control-") or name in {
            f"{PREFIX}splits.json",
            f"{PREFIX}test_subsplits.json",
        }:
            objects.append({
                "name": name,
                "generation": item["generation"],
                "size": int(item["size"]),
                "md5_base64": item["md5Hash"],
                "media_link": item["mediaLink"],
            })
    return sorted(objects, key=lambda item: item["name"])


def md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def download_object(item: dict, output_dir: Path) -> dict:
    output = output_dir / Path(item["name"]).name
    partial = output.with_suffix(output.suffix + ".partial")
    if output.is_file() and output.stat().st_size == item["size"]:
        if md5_base64(output) != item["md5_base64"]:
            raise ValueError(f"existing object MD5 mismatch: {output}")
        return {**item, "path": str(output), "status": "VERIFIED_EXISTING"}

    for attempt in range(20):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["size"]:
            raise ValueError(f"partial object exceeds expected size: {partial}")
        if offset == item["size"]:
            break
        request = urllib.request.Request(item["media_link"])
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(request) as response, partial.open(
            "ab" if offset else "wb"
        ) as handle:
            if offset and response.status != 206:
                raise ValueError(f"server ignored Range request for {item['name']}")
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        raise ValueError(f"download retry limit reached: {item['name']}")
    if partial.stat().st_size != item["size"]:
        raise ValueError(
            f"size mismatch for {item['name']}: {partial.stat().st_size} != {item['size']}"
        )
    actual_md5 = md5_base64(partial)
    if actual_md5 != item["md5_base64"]:
        raise ValueError(f"downloaded object MD5 mismatch: {item['name']}")
    partial.replace(output)
    return {**item, "path": str(output), "status": "DOWNLOADED_MD5_VERIFIED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--small-files-only", action="store_true")
    args = parser.parse_args()

    objects = fetch_objects()
    if len(objects) != 22:
        raise ValueError(f"expected 20 shards and 2 metadata files, found {len(objects)}")
    selected = objects
    if args.small_files_only:
        selected = [item for item in objects if item["name"].endswith(".json")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        completed = [{**item, "status": "LISTED"} for item in selected]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            completed = list(executor.map(
                lambda item: download_object(item, args.output_dir), selected
            ))
    manifest = {
        "status": "LISTED" if args.metadata_only else "DOWNLOADED_MD5_VERIFIED",
        "bucket": BUCKET,
        "prefix": PREFIX,
        "objects": completed,
        "total_bytes": sum(item["size"] for item in completed),
    }
    manifest_path = args.output_dir / (
        "small_files_manifest.json" if args.small_files_only else "official_gcs_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "objects": len(completed),
        "total_bytes": manifest["total_bytes"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()