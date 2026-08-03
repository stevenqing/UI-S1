import argparse
import json
import math
from pathlib import Path

from common import UPSTREAM, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flips = read_json(UPSTREAM / "w2_flips.json")
    original = read_json(UPSTREAM / "w2_noise.json")
    cells = {
        **flips["androidcontrol"]["cells"],
        **flips["mind2web"]["cells"],
    }
    representatives = (
        ("androidcontrol", "gui-r1-7b/low"),
        ("androidcontrol", "gui-r1-7b/high"),
        ("androidcontrol", "ui-agile-7b/low"),
        ("androidcontrol", "ui-agile-7b/high"),
        ("mind2web", "tongui-7b/visual"),
    )
    output = {}
    for bench, key in representatives:
        full = cells[f"{key}/full"]["step_sr"]
        v1 = cells[f"{key}/v1"]["step_sr"]
        delta = v1 - full
        original_noise = original[bench][key]
        output[f"{bench}/{key}"] = {
            "full_step_sr": full,
            "v1_step_sr": v1,
            "signed_delta": delta,
            "absolute_delta": abs(delta),
            "sample_sd_full_v1": abs(delta) / math.sqrt(2),
            "mde_v1_only": math.sqrt(2) * abs(delta),
            "original_five_view_sample_sd": original_noise["sample_sd"],
            "original_five_view_mde": original_noise["mde"],
        }
    result = {
        "status": "PASS",
        "definition": "MDE_v1 = 2 * sample SD(full,v1) = sqrt(2) * abs(v1-full)",
        "main_exchangeable_views": ["full", "v1"],
        "excluded_distribution_shift_views": ["v2", "v3", "v4"],
        "cells": output,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
