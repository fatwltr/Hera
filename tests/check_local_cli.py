#!/usr/bin/env python3
"""Require complete, independently parseable reports from both local parties."""

import subprocess
import sys
import json
import math
from pathlib import Path


def main():
    result = subprocess.run(
        sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    sys.stdout.write(result.stdout)
    lines = result.stdout.splitlines()
    config = next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--config="))
    block_count = len(json.loads(Path(config).read_text())["model"]["layers"])
    blocks = [line for line in lines if line.startswith("HERA_BLOCK=")]
    totals = [line for line in lines if line.startswith("HERA_TOTAL,")]
    metrics_valid = True
    expected_metrics = {"HE_MS", "MPC_MS", "CONVERSION_MS", "HE_COMM_BYTES",
                        "MPC_COMM_BYTES", "CONVERSION_COMM_BYTES"}
    for line in blocks + totals:
        try:
            values = {key: float(value) for key, value in
                      (field.split("=", 1) for field in line.split(",")[1:])}
            metrics_valid &= (
                values.keys() == expected_metrics
                and all(math.isfinite(value) and value >= 0 for value in values.values())
                and values["HE_COMM_BYTES"] == 0
                and values["MPC_COMM_BYTES"] > 0
                and values["CONVERSION_COMM_BYTES"] > 0
            )
        except (ValueError, KeyError):
            metrics_valid = False
    valid = (
        result.returncode == 0
        and lines.count("HERA_PRIVATE_INFERENCE_STATUS=PASS") == 2
        and lines.count("HERA_PARTY=server") == 1
        and lines.count("HERA_PARTY=client") == 1
        and len(blocks) == block_count * 2
        and len(totals) == (2 if block_count > 1 else 0)
        and metrics_valid
        and all(line.startswith(("HERA_PARTY=", "HERA_BLOCK=", "HERA_TOTAL,",
                                 "HERA_PRIVATE_INFERENCE_STATUS=")) for line in lines)
        and "STATUS=FAIL" not in result.stdout
    )
    if not valid:
        print("HERA_LOCAL_CLI_STATUS=FAIL: expected two concise online measurement reports")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
