import argparse
import hashlib
import importlib.util
import subprocess
import tempfile
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UPSTREAM_SCORING = ROOT / "runs/complementarity/2026-07-30/scoring.py"
EXPECTED_SUMMARY_SHA256 = {
    "androidcontrol": "5c4e9495c1b1eaaee46fad7101ef148174bb41d1cf51c5b99b4931ff2188adfb",
    "mind2web": "f0418b3f42806ac6026cadc7e35ad4c948128256c20294798e1e3ff21baa6609",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_scoring():
    spec = importlib.util.spec_from_file_location("collision_upstream_scoring", UPSTREAM_SCORING)
    if spec is None or spec.loader is None:
        raise ImportError(UPSTREAM_SCORING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = load_upstream_scoring()
ACTION_TO_ID = UPSTREAM.ACTION_TO_ID
GROUNDING_ACTIONS = UPSTREAM.GROUNDING_ACTIONS
TEXT_ACTIONS = UPSTREAM.TEXT_ACTIONS
SIMPLE_ACTIONS = UPSTREAM.SIMPLE_ACTIONS
token_f1 = UPSTREAM.token_f1
text_f1 = UPSTREAM.text_f1
label_android_row = UPSTREAM.label_android_row
android_metric_counts = UPSTREAM.android_metric_counts
score_mind2web_row = UPSTREAM.score_mind2web_row
distance_to_bbox = UPSTREAM.distance_to_bbox
normalized_bbox = UPSTREAM.normalized_bbox


def verify_locked_summaries() -> None:
    paths = {
        "androidcontrol": ROOT / "runs/error-overlap-analysis/2026-07-29/androidcontrol_summary.json",
        "mind2web": ROOT / "runs/error-overlap-analysis/2026-07-29/mind2web_summary.json",
    }
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED_SUMMARY_SHA256[name]:
            raise ValueError(f"locked {name} summary hash mismatch: {actual}")


def run_bytewise_regression() -> dict:
    verify_locked_summaries()
    original = {
        "androidcontrol": ROOT / "runs/error-overlap-analysis/2026-07-29/androidcontrol_summary.json",
        "mind2web": ROOT / "runs/error-overlap-analysis/2026-07-29/mind2web_summary.json",
    }
    with tempfile.TemporaryDirectory(prefix="collision-scoring-") as directory:
        temporary = Path(directory)
        regenerated = {
            "androidcontrol": temporary / "androidcontrol_summary.json",
            "mind2web": temporary / "mind2web_summary.json",
        }
        subprocess.run([
            str(ROOT / ".venv-ac-vllm/bin/python"),
            str(ROOT / "runs/error-overlap-analysis/2026-07-29/analyze_androidcontrol.py"),
            "--output", str(regenerated["androidcontrol"]),
        ], cwd=ROOT, check=True)
        subprocess.run([
            str(ROOT / "runs/mindact/2026-07-29/run_python.sh"),
            str(ROOT / "runs/error-overlap-analysis/2026-07-29/analyze_mind2web.py"),
            "--output", str(regenerated["mind2web"]),
        ], cwd=ROOT, check=True)
        result = {}
        for name in original:
            original_bytes = original[name].read_bytes()
            regenerated_bytes = regenerated[name].read_bytes()
            if original_bytes != regenerated_bytes:
                raise ValueError(f"{name} scoring regression is not byte-identical")
            result[name] = {
                "status": "PASS",
                "sha256": hashlib.sha256(original_bytes).hexdigest(),
                "bytes": len(original_bytes),
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()
    if not args.regression:
        parser.error("--regression is required for the command-line interface")
    print(run_bytewise_regression())


if __name__ == "__main__":
    main()