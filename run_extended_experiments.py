"""Compatibility entry point for the current uniform-20 study."""
import sys

from run_uniform20_experiments import cli, main as main

if __name__ == "__main__":
    sys.exit(cli())
