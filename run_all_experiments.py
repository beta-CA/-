"""Right-click Run this file in PyCharm to train/resume all six experiments.

No script parameters are needed. This delegates to the existing verified runner
with --execute --resume. For a safe preview, pass --dry-run.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parent
    script = root / "run_experiments.py"
    os.chdir(root)
    sys.argv = [str(script), "--execute", "--resume", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
