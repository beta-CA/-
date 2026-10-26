"""Right-click Run to execute/resume the fixed approximately-three-hour extension.

The original six-run study and its results remain unchanged. --dry-run prints
the added work and measured-rate budget without data access or training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import extended_protocol as protocol


def training_command(root, output, spec):
    command = [sys.executable, "-u", str(root / "train_extended.py"),
               "--case-id", spec["run_id"], "--project-root", str(root),
               "--output-root", str(output), "--data-root", str(root / "data"),
               "--device", "cuda", "--workers", "2"]
    if spec["warm_start_run_id"]:
        command += ["--warm-start-checkpoint", str(root / "runs" / spec["warm_start_run_id"] / "last.pth")]
    if (output / spec["run_id"] / "last.pth").is_file():
        command.append("--resume")
    return command


def run_logged(command, root, log_path):
    """Keep the traceback on disk while retaining live PyCharm output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {subprocess.list2cmdline(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    manifest = protocol.verify_manifest(root)
    output = root / "runs_extended"
    results = root / "results_extended"
    print("Extended study: 3-seed main comparison; 2-seed temperature and alpha ablations.", flush=True)
    print("19 new/continued training tasks plus 2 reused teachers; all students have a 20-epoch budget.", flush=True)
    budget = protocol.estimated_budget(root)
    print(json.dumps(budget, indent=2), flush=True)
    commands = []
    for spec in protocol.training_cases():
        command = training_command(root, output, spec)
        commands.append((spec, command))
        print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        print("PREVIEW ONLY: no preparation, training or evaluation was started.")
        print("Default T=2/alpha=.25 cases are reused across main and ablation tables.")
        return

    from kd_utils import file_sha256, load_json, save_json
    from prepare_data import prepare_data
    from summarize import audit_matrix as audit_original
    from train_extended import validate_run
    from extended_recovery import recover_initialization
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    def status(phase, **extra):
        save_json(reports / "extended_pipeline_status.json", {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
            "study": protocol.STUDY, **extra})

    status("preflight")
    audit_original(root / "runs")
    original_paths = [root / name for name in ("train.py", "kd_utils.py", "models.py", "evaluate.py",
                      "summarize.py", "configs/experiments.json")]
    original_paths += [path for folder in (root / "runs", root / "results")
                       for path in folder.rglob("*") if path.is_file()]
    preserved = {path.relative_to(root).as_posix(): file_sha256(path) for path in original_paths}
    preflight_path = reports / "extended_preflight.json"
    if preflight_path.exists():
        prior = load_json(preflight_path)
        if prior["original_files_sha256"] != preserved:
            raise RuntimeError("Original study changed since extension preflight; inspect before resuming")
    else:
        save_json(preflight_path, {"study": protocol.STUDY,
                  "original_files_sha256": preserved, "budget": budget,
                  "protocol": manifest, "exploratory_extension": True})
    prepare_data(root)
    for index, (spec, _) in enumerate(commands, 1):
        case_id = spec["run_id"]
        log_path = reports / "extended_logs" / f"{case_id}.log"
        try:
            recovery = recover_initialization(case_id, root, output)
            if recovery["recovered"]:
                save_json(reports / f"extended_recovery_{case_id}.json", recovery)
                print(f"Recovered saved initialization for {case_id} at epoch {recovery['epoch']}.", flush=True)
            folder = output / case_id
            if (folder / "summary.json").is_file() and load_json(folder / "summary.json").get("status") == "completed":
                validate_run(folder / "best.pth")
                print(f"Already completed, verified and skipped: {case_id}", flush=True)
                continue
            status("training", index=index, total=19, case_id=case_id, log_file=str(log_path))
            print(f"\nExtended training {index}/19: {case_id}", flush=True)
            for attempt in range(2):
                command = training_command(root, output, spec)
                code = run_logged(command, root, log_path)
                if code == 0:
                    break
                recovery = recover_initialization(case_id, root, output)
                if attempt == 0 and recovery["recovered"]:
                    save_json(reports / f"extended_recovery_{case_id}.json", recovery)
                    print(f"Recovered Windows initialization publication failure; resuming {case_id}.", flush=True)
                    continue
                raise subprocess.CalledProcessError(code, command)
        except Exception as error:
            status("failed", index=index, total=19, case_id=case_id,
                   log_file=str(log_path), error=str(error))
            raise
    for spec, _ in commands:
        validate_run(output / spec["run_id"] / "best.pth")
    status("evaluating")
    command = [sys.executable, "-u", str(root / "evaluate_extended.py"),
                    "--project-root", str(root), "--runs-root", str(output),
                    "--output-root", str(results), "--device", "cuda", "--workers", "2"]
    code = run_logged(command, root, reports / "extended_logs" / "evaluation.log")
    if code:
        status("failed", stage="evaluating", returncode=code,
               log_file=str(reports / "extended_logs" / "evaluation.log"))
        raise subprocess.CalledProcessError(code, command)
    status("summarizing")
    command = [sys.executable, "-u", str(root / "summarize_extended.py"),
                    "--project-root", str(root), "--runs-root", str(output),
                    "--output-root", str(results)]
    code = run_logged(command, root, reports / "extended_logs" / "summarize.log")
    if code:
        status("failed", stage="summarizing", returncode=code,
               log_file=str(reports / "extended_logs" / "summarize.log"))
        raise subprocess.CalledProcessError(code, command)
    if any(file_sha256(root / name) != digest for name, digest in preserved.items()):
        raise RuntimeError("Original study files changed during execution; inspect preservation audit")
    status("completed", training_tasks=19, unique_cases=21, original_files_unchanged=True)
    print(f"Extension complete. Weights/logs: {output}; tables/figures: {results}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Run this same entry again to resume from committed epoch boundaries.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Extended study stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
