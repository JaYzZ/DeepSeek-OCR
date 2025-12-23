"""
Minimal SynthDoG runner for quick synthetic samples.

It shells out to synthtiger with the upstream template/config shipped in
../donut/synthdog. By default it generates 5 English samples into
./Donut/synthdog_outputs. Override the count via SYNTHDOG_COUNT env var.
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    donut_repo = here.parent / "donut"
    synthdog_dir = donut_repo / "synthdog"
    template = synthdog_dir / "template.py"
    config = synthdog_dir / "config_en.yaml"

    if not template.is_file() or not config.is_file():
        raise SystemExit("SynthDoG template/config not found under ../donut/synthdog")

    out_dir = here / "synthdog_outputs"
    count = os.environ.get("SYNTHDOG_COUNT", "5")
    workers = os.environ.get("SYNTHDOG_WORKERS", "2")

    cmd = [
        "synthtiger",
        "-o",
        str(out_dir),
        "-c",
        str(count),
        "-w",
        str(workers),
        "-v",
        str(template),
        "SynthDoG",
        str(config),
    ]

    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "synthtiger CLI not found. Install it first (pip install synthtiger)."
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"synthtiger failed with exit code {exc.returncode}") from exc

    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
