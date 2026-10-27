"""Run/resume 21 independent CIFAR-10 cases, all with a 20-epoch budget.

Right-click Run in PyCharm. --dry-run previews commands without training.
Historical ten-epoch teachers and their students are never reused.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import uniform_protocol as protocol
from run_legacy_extended_experiments import run_logged


def training_command(root, output, spec):
    command = [sys.executable, "-u", str(root / "train_uniform20.py"),
               "--case-id", spec["run_id"], "--project-root", str(root),
               "--output-root", str(output), "--data-root", str(root / "data"),
               "--device", "cuda", "--workers", "2"]
    if (output / spec["run_id"] / "last.pth").is_file():
        command.append("--resume")
    return command


def historical_hashes(root):
    from kd_utils import file_sha256
    paths = [root / name for name in (
        "train.py", "models.py", "kd_utils.py", "evaluate.py", "summarize.py",
        "train_extended.py", "extended_protocol.py", "evaluate_extended.py",
        "summarize_extended.py", "extended_recovery.py", "configs/experiments.json",
        "configs/extended_experiments.json")]
    for name in ("runs", "results", "runs_extended", "results_extended"):
        folder = root / name
        if folder.exists():
            paths.extend(p for p in folder.rglob("*") if p.is_file()
                         and not any(part.startswith(".") for part in p.relative_to(folder).parts))
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    manifest = protocol.verify_manifest(root)
    output, results = root / protocol.RUNS_DIRECTORY, root / protocol.RESULTS_DIRECTORY
    specs = protocol.training_cases()
    total = len(specs)
    print(f"Uniform-20 study: {total} cases x {protocol.TRAINING_EPOCHS} epochs = "
          f"{sum(s['epochs'] for s in specs)} training epochs.", flush=True)
    print("3 teachers + 3 CE students + 15 KD students. All start fresh in the new study.", flush=True)
    print(f"Weights/logs: {output}\nTables/Chinese figures: {results}", flush=True)
    budget = protocol.estimated_budget(root)
    print(json.dumps(budget, indent=2), flush=True)
    for spec in specs:
        print(subprocess.list2cmdline(training_command(root, output, spec)), flush=True)
    if args.dry_run:
        print("PREVIEW ONLY: no data preparation, training, evaluation or output creation.")
        return

    from kd_utils import load_json, save_json
    from prepare_data import prepare_data
    from train_uniform20 import validate_run
    from uniform20_recovery import recover_initialization

    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    log_root = reports / "uniform20_logs"

    def status(phase, **details):
        save_json(reports / "uniform20_pipeline_status.json", {
            "study": protocol.STUDY, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase, **details})

    status("preflight")
    preserved = historical_hashes(root)
    preflight = reports / "uniform20_preflight.json"
    if preflight.is_file():
        prior = load_json(preflight)
        if prior.get("protocol") != manifest or prior.get("historical_files_sha256") != preserved:
            raise RuntimeError("Protocol or historical artifacts changed since this study started; inspect before resume")
    else:
        save_json(preflight, {"study": protocol.STUDY, "protocol": manifest,
                             "historical_files_sha256": preserved, "budget": budget})
    local_paths = root / "configs" / "local_paths.json"
    reuse = load_json(local_paths).get("reuse_data_root") if local_paths.is_file() else None
    status("preparing_data")
    prepare_data(root, Path(reuse) if reuse else None)

    for index, spec in enumerate(specs, 1):
        case_id = spec["run_id"]
        log_path = log_root / f"{case_id}.log"
        try:
            recover_initialization(case_id, root, output)
            folder = output / case_id
            if (folder / "summary.json").is_file() and load_json(folder / "summary.json").get("status") == "completed":
                validate_run(folder / "best.pth")
                print(f"Verified completed 20-epoch case, skipped: {case_id}", flush=True)
                continue
            status("training", index=index, total=total, case_id=case_id, log_file=str(log_path))
            print(f"\nUniform-20 training {index}/{total}: {case_id}", flush=True)
            for attempt in range(2):
                command = training_command(root, output, spec)
                code = run_logged(command, root, log_path)
                if code == 0:
                    validate_run(output / case_id / "best.pth")
                    break
                recovery = recover_initialization(case_id, root, output)
                if attempt == 0 and recovery["recovered"]:
                    print(f"Recovered Windows initialization for {case_id}; resuming.", flush=True)
                    continue
                raise subprocess.CalledProcessError(code, command)
        except Exception as error:
            status("failed", index=index, total=total, case_id=case_id,
                   log_file=str(log_path), error=str(error))
            raise

    for spec in specs:
        validate_run(output / spec["run_id"] / "best.pth")
    for phase, script, extra in (
        ("evaluating", "evaluate_uniform20.py", ["--device", "cuda", "--workers", "2"]),
        ("summarizing", "summarize_uniform20.py", []),
    ):
        status(phase)
        command = [sys.executable, "-u", str(root / script), "--project-root", str(root),
                   "--runs-root", str(output), "--output-root", str(results), *extra]
        log_path = log_root / f"{phase}.log"
        code = run_logged(command, root, log_path)
        if code:
            status("failed", stage=phase, returncode=code, log_file=str(log_path))
            raise subprocess.CalledProcessError(code, command)
    if historical_hashes(root) != preserved:
        raise RuntimeError("Historical experiment artifacts changed during execution")
    status("completed", unique_cases=total, epochs_each=protocol.TRAINING_EPOCHS,
           total_training_epochs=sum(s["epochs"] for s in specs), historical_files_unchanged=True)
    print(f"Uniform-20 complete. Weights/logs: {output}; tables/Chinese figures: {results}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Run this same file to resume from completed epoch boundaries.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Uniform-20 study stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
