import importlib.util
import json
import math
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UPSTREAM = ROOT / "runs/collision-law/2026-07-30"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def finite_mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return None
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - spread, center + spread]
