"""Current entry: all teachers and students now train for 20 epochs.

Right-click Run; --dry-run previews. Historical reproduction explicitly uses
run_legacy_extended_experiments.py instead.
"""
import sys
from run_uniform20_experiments import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Run this file again to resume the uniform-20 study.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Uniform-20 study stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
