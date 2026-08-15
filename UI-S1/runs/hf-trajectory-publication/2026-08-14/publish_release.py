import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


RUN_DIR = Path(__file__).resolve().parent
STAGE_ROOT = Path("/scratch/workspaceblobstore/hf-trajectory-publication/2026-08-14")
PACKAGE_MANIFEST = RUN_DIR / "PACKAGE_MANIFEST.json"
STATUS_PATH = RUN_DIR / "PUBLICATION_STATUS.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_package(package, expected):
    release_path = package / "RELEASE_MANIFEST.json"
    readme_path = package / "README.md"
    if sha256_file(release_path) != expected["release_manifest_sha256"]:
        raise ValueError(f"release manifest changed: {package.name}")
    if sha256_file(readme_path) != expected["readme_sha256"]:
        raise ValueError(f"README changed: {package.name}")
    release = json.loads(release_path.read_text())
    for relative, info in release["artifacts"].items():
        path = package / relative
        if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
            raise ValueError(f"staged artifact changed: {package.name}/{relative}")
    return release


def main():
    if STATUS_PATH.exists():
        raise FileExistsError(STATUS_PATH)
    package_manifest = json.loads(PACKAGE_MANIFEST.read_text())
    if package_manifest["status"] != "PASS_HF_TRAJECTORY_PACKAGES_VALIDATED":
        raise ValueError("package manifest is not authorized")
    api = HfApi()
    username = api.whoami()["name"]
    results = {}
    for benchmark, expected in package_manifest["releases"].items():
        package = STAGE_ROOT / benchmark
        release = verify_local_package(package, expected)
        repo_id = release["repo_id"]
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if not info.private:
            raise ValueError(f"repository is not private: {repo_id}")
        commit = api.upload_folder(
            folder_path=str(package),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Publish verified UI-S1 {benchmark} trajectories",
        )
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=commit.oid))
        expected_files = set(release["artifacts"]) | {"README.md", "RELEASE_MANIFEST.json"}
        if not expected_files.issubset(remote_files):
            raise ValueError(f"remote files missing for {repo_id}: {sorted(expected_files - remote_files)}")
        with tempfile.TemporaryDirectory() as temporary:
            remote_manifest = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="RELEASE_MANIFEST.json",
                revision=commit.oid,
                local_dir=temporary,
            )
            remote_readme = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="README.md",
                revision=commit.oid,
                local_dir=temporary,
            )
            if sha256_file(remote_manifest) != expected["release_manifest_sha256"]:
                raise ValueError(f"remote manifest mismatch: {repo_id}")
            if sha256_file(remote_readme) != expected["readme_sha256"]:
                raise ValueError(f"remote README mismatch: {repo_id}")
        results[benchmark] = {
            "repo_id": repo_id,
            "private": True,
            "revision": commit.oid,
            "remote_file_count": len(remote_files),
            "expected_file_count": len(expected_files),
            "artifact_count": release["artifact_count"],
            "total_rows": release["total_rows"],
            "total_bytes": release["total_bytes"],
            "release_manifest_sha256": expected["release_manifest_sha256"],
            "readme_sha256": expected["readme_sha256"],
            "remote_verified": True,
        }
    status = {
        "schema_version": 1,
        "status": "COMPLETE_HF_PRIVATE_TRAJECTORY_PUBLICATION",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "authenticated_user": username,
        "visibility": "private",
        "ground_truth_uploaded": False,
        "private_labels_uploaded": False,
        "benchmark_images_uploaded": False,
        "releases": results,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS_PATH)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()