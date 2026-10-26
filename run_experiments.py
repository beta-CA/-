"""Preview or explicitly execute the six-run knowledge-distillation study."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def load_matrix():
    matrix = json.loads((ROOT / "configs" / "experiments.json").read_text(encoding="utf-8"))
    expected = [(role, seed) for seed in (0, 1) for role in ("teacher", "baseline", "kd")]
    if [(item["role"], item["seed"]) for item in matrix["runs"]] != expected:
        raise ValueError("The frozen sequence must be teacher/baseline/KD for seeds 0 and 1")
    fixed = {"epochs": 10, "batch_size": 128, "workers": 2, "device": "cuda",
             "precision": "float32", "learning_rate": 0.001, "temperature": 2.0,
             "alpha": 0.25, "split_seed": 20261026, "extra_augmentation": False}
    for key, value in fixed.items():
        if matrix["protocol"].get(key) != value:
            raise ValueError(f"Frozen protocol changed: {key}")
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually prepare data, train, evaluate and summarize")
    parser.add_argument("--resume", action="store_true", help="Resume runs with last.pth; completed runs are not retrained")
    parser.add_argument("--dry-run", action="store_true", help="Print only; overrides --execute")
    args = parser.parse_args()
    matrix = load_matrix()
    commands = []
    for item in matrix["runs"]:
        role, seed = item["role"], item["seed"]
        run_dir = ROOT / "runs" / f"{role}_seed{seed}"
        command = [sys.executable, "-u", str(ROOT / "train.py"), "--role", role,
                   "--seed", str(seed), "--epochs", "10", "--data-root", str(ROOT / "data"),
                   "--output-root", str(ROOT / "runs"), "--workers", "2", "--device", "cuda"]
        if role == "kd":
            command += ["--teacher-checkpoint", str(ROOT / "runs" / f"teacher_seed{seed}" / "best.pth")]
        if args.resume and (run_dir / "last.pth").is_file():
            command.append("--resume")
        commands.append((item, command))
    print("Frozen study: 2 seeds x (teacher + baseline student + KD student), 10 epochs each.", flush=True)
    for item, command in commands:
        print(subprocess.list2cmdline(command), flush=True)
    if not args.execute or args.dry_run:
        print("PREVIEW ONLY. No data preparation or training. Use --execute --resume, or the PyCharm Run All configuration.")
        print("After all six complete: final evaluation of every best.pth, then complete tables/figures.")
        return

    from prepare_data import prepare_data
    from kd_utils import load_json, save_json
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    def status(phase, **details):
        save_json(reports / "pipeline_status.json", {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, **details})
    status("preparing_data")
    local_path = ROOT / "configs" / "local_paths.json"
    reuse = load_json(local_path).get("reuse_data_root") if local_path.exists() else None
    prepare_data(ROOT, Path(reuse) if reuse else None)
    for index, (item, command) in enumerate(commands, 1):
        status("training", index=index, total=6, **item)
        print(f"\nTraining {index}/6: {item['role']}, seed={item['seed']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    # No official test evaluation until every planned training run is complete.
    for item, _ in commands:
        folder = ROOT / "runs" / f"{item['role']}_seed{item['seed']}"
        summary = load_json(folder / "summary.json")
        config = load_json(folder / "config.json")
        if (summary.get("status") != "completed" or summary.get("completed_epochs") != 10
                or summary.get("requested_epochs") != 10 or config.get("dataset") != "CIFAR10"):
            raise RuntimeError(f"Run is not a completed formal 10-epoch experiment: {folder}")
        if (config.get("role"), config.get("seed")) != (item["role"], item["seed"]):
            raise RuntimeError(f"Run identity differs from matrix: {folder}")
    for index, (item, _) in enumerate(commands, 1):
        checkpoint = ROOT / "runs" / f"{item['role']}_seed{item['seed']}" / "best.pth"
        status("evaluating", index=index, total=6, **item)
        subprocess.run([sys.executable, "-u", str(ROOT / "evaluate.py"), "--checkpoint", str(checkpoint),
                        "--data-root", str(ROOT / "data"), "--device", "cuda", "--workers", "2"],
                       cwd=ROOT, check=True)
    status("summarizing")
    subprocess.run([sys.executable, "-u", str(ROOT / "summarize.py"),
                    "--runs-root", str(ROOT / "runs"), "--output-root", str(ROOT / "results")],
                   cwd=ROOT, check=True)
    status("completed", runs=6, epochs_each=10)
    print(f"Complete. Weights/logs: {ROOT / 'runs'}; tables/figures: {ROOT / 'results'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Re-run the same configuration to resume from the last complete epoch.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
