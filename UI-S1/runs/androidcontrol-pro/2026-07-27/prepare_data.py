import argparse
import base64
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "data/official_test_mirror/data"
AC_IDX = ROOT.parent.parent / "androidcontrol/2026-07-27/repos/OS-Atlas/eval/data/ac_idx.txt"
OS_ATLAS_SAMPLE = ROOT.parent.parent / "androidcontrol/2026-07-27/repos/OS-Atlas/eval/data/ac_test.jsonl"


def format_action(action, width, height):
    action_type = action["action_type"]
    if action_type == "click":
        return f'CLICK <point>[[{int(action["x"] / width * 1000)},{int(action["y"] / height * 1000)}]]</point>'
    if action_type == "long_press":
        return f'LONG_PRESS <point>[[{int(action["x"] / width * 1000)},{int(action["y"] / height * 1000)}]]</point>'
    if action_type == "input_text":
        return f'TYPE [{action["text"]}]'
    if action_type == "scroll":
        return f'SCROLL [{action["direction"].upper()}]'
    if action_type == "open_app":
        return f'OPEN_APP [{action["app_name"]}]'
    if action_type == "navigate_back":
        return "PRESS_BACK"
    if action_type == "navigate_home":
        return "PRESS_HOME"
    if action_type == "wait":
        return "WAIT"
    raise ValueError(f"unsupported action type: {action_type}")


def decode_png(encoded):
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if image.format != "PNG":
            raise ValueError(f"expected PNG, found {image.format}")
    return image_bytes, width, height


def history_text(step_instructions, step_id):
    if step_id == 0:
        return "None"
    return "\n".join(
        f"Step {index + 1}: {instruction}"
        for index, instruction in enumerate(step_instructions[:step_id])
    )


def sample_actions():
    actions = []
    with OS_ATLAS_SAMPLE.open() as sample_file:
        for line in sample_file:
            row = json.loads(line)
            actions.append(row["conversations"][1]["value"].split("actions:\n", 1)[1])
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    identities = AC_IDX.read_text().splitlines()
    if args.limit is not None:
        identities = identities[: args.limit]
    selected = set(identities)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    records = {}

    for parquet_path in sorted(PARQUET_DIR.glob("test-*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["episode_id", "goal", "screenshots_b64", "actions", "step_instructions"],
        )
        for episode in table.to_pylist():
            episode_id = episode["episode_id"]
            for step_id, action in enumerate(episode["actions"]):
                identity = f"episode_{episode_id}_screenshot_{step_id}"
                if identity not in selected:
                    continue
                image_bytes, width, height = decode_png(episode["screenshots_b64"][step_id])
                image_path = images_dir / f"{identity}.png"
                image_path.write_bytes(image_bytes)
                records[identity] = {
                    "identity": identity,
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "goal": episode["goal"],
                    "history": history_text(episode["step_instructions"], step_id),
                    "low_instruction_audit_only": episode["step_instructions"][step_id],
                    "image": str(image_path.relative_to(ROOT)),
                    "image_width": width,
                    "image_height": height,
                    "gt_action": format_action(action, width, height),
                    "source_action": action,
                }

    missing = sorted(selected - set(records))
    if missing:
        raise RuntimeError(f"missing {len(missing)} selected identities: {missing[:10]}")
    output_path = args.output_dir / "ac_high.jsonl"
    with output_path.open("w") as output_file:
        for identity in identities:
            output_file.write(json.dumps(records[identity], ensure_ascii=False) + "\n")

    if identities[:3] == [
        "episode_140_screenshot_0",
        "episode_140_screenshot_1",
        "episode_140_screenshot_2",
    ]:
        expected = sample_actions()
        actual = [records[identity]["gt_action"] for identity in identities[:3]]
        if actual != expected:
            raise RuntimeError(f"OS-Atlas sample action mismatch: {actual} != {expected}")

    summary = {
        "status": "PASS",
        "rows": len(records),
        "images": len(list(images_dir.glob("*.png"))),
        "missing": len(missing),
        "first_identity": identities[0],
        "last_identity": identities[-1],
        "output": str(output_path.relative_to(ROOT)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()