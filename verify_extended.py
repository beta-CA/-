"""Verify only the extension's new behavior, using tagged synthetic fixtures.

No original experiment is changed and no official test images are loaded.
The short fixtures are not classification results and cannot enter formal tables.
Run with --device cuda --workers 2 to cover the Windows loader configuration.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch
from torch.nn import functional as F

import extended_protocol as protocol
from kd_utils import file_sha256, load_checkpoint, load_json, model_state_sha256
from models import build_model, distillation_loss

ROOT = Path(__file__).resolve().parent


def equal_nested(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            equal_nested(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            equal_nested(a, b) for a, b in zip(first, second))
    return first == second


def metrics_without_time(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: value for key, value in row.items() if key != "seconds"}
                for row in csv.DictReader(stream)]


def tree_hashes(folder):
    return {file.relative_to(folder).as_posix(): file_sha256(file)
            for file in Path(folder).rglob("*") if file.is_file()
            and "__pycache__" not in file.parts}


def frozen_source_hashes():
    files = [ROOT / name for name in ("train.py", "kd_utils.py", "models.py",
                                      "evaluate.py", "summarize.py", "configs/experiments.json")]
    files += [file for file in (ROOT / "vendor").rglob("*")
              if file.is_file() and "__pycache__" not in file.parts]
    return {file.relative_to(ROOT).as_posix(): file_sha256(file) for file in files}


def verify_protocol():
    cases, all_cases = protocol.training_cases(), protocol.all_cases()
    assert len(cases) == 19 and len(all_cases) == 21
    assert len({item["run_id"] for item in cases}) == 19
    assert len({item["run_id"] for item in all_cases}) == 21
    legacy = [item for item in all_cases if item["reuse_legacy_teacher"]]
    assert {item["run_id"] for item in legacy} == {"teacher_seed0", "teacher_seed1"}
    assert all(item["role"] == "teacher" and item["epochs"] == 10 for item in legacy)
    warm = [item for item in cases if item["warm_start_run_id"]]
    assert {item["run_id"] for item in warm} == {
        "baseline_seed0", "baseline_seed1", "kd_seed0", "kd_seed1"}
    new_epochs = {"teacher": 0, "baseline": 0, "kd": 0}
    for item in cases:
        new_epochs[item["role"]] += item["epochs"] - (10 if item["warm_start_run_id"] else 0)
    assert new_epochs == {"teacher": 10, "baseline": 40, "kd": 280}
    for seed in protocol.ABLATION_SEEDS:
        default = f"kd_seed{seed}"
        assert protocol.temperature_case_id(2, seed) == default
        assert protocol.alpha_case_id(.25, seed) == default
        for value in protocol.TEMPERATURES:
            item = protocol.case_spec(protocol.temperature_case_id(value, seed))
            assert item["temperature"] == value and item["alpha"] == .25 and item["epochs"] == 20
        for value in protocol.ALPHAS:
            item = protocol.case_spec(protocol.alpha_case_id(value, seed))
            assert item["alpha"] == value and item["temperature"] == 2 and item["epochs"] == 20
    assert all(item["epochs"] == 20 for item in cases if item["role"] != "teacher")
    for seed in protocol.MAIN_SEEDS:
        assert all(protocol.case_spec(f"{role}_seed{seed}")["seed"] == seed
                   for role in ("teacher", "baseline", "kd"))
    return {"predeclared_19_new_21_unique_cases": True,
            "default_shared_without_redundant_ablation_training": True,
            "new_epoch_budget_40_baseline_280_kd_10_teacher": True,
            "all_student_ablation_budgets_twenty_epochs": True}


def verify_parameterized_objective():
    generator = torch.Generator().manual_seed(814)
    initial = torch.randn(7, 10, dtype=torch.float64, generator=generator)
    teacher = torch.randn(7, 10, dtype=torch.float64, generator=generator, requires_grad=True)
    labels = torch.arange(7) % 10
    for temperature in protocol.TEMPERATURES:
        for alpha in protocol.ALPHAS:
            logits = initial.clone().requires_grad_()
            reference_logits = initial.clone().requires_grad_()
            total, ce, kd = distillation_loss(logits, teacher, labels, temperature, alpha)
            targets = F.softmax(teacher.detach() / temperature, dim=-1)
            log_student = F.log_softmax(reference_logits / temperature, dim=-1)
            reference_kd = torch.sum(targets * (targets.log() - log_student)) / len(labels) * temperature ** 2
            reference_ce = F.cross_entropy(reference_logits, labels)
            reference_total = (1 - alpha) * reference_ce + alpha * reference_kd
            for actual, expected in ((total, reference_total), (ce, reference_ce), (kd, reference_kd)):
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            total.backward()
            reference_total.backward()
            torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-11, atol=1e-12)
            assert teacher.grad is None
    return {"temperature_weight_grid_loss_and_gradient_equivalence": True,
            "teacher_detached_for_every_declared_hyperparameter": True}


def compare_boundaries(first_folder, second_folder, compare_config=False):
    first = load_checkpoint(Path(first_folder) / "last.pth")
    second = load_checkpoint(Path(second_folder) / "last.pth")
    fields = ["epoch", "model_state", "model_state_sha256", "optimizer_state", "rng_state",
              "init_sha256", "best_epoch", "best_val_acc", "best_val_loss", "best_model_state_sha256"]
    if compare_config:
        fields.append("config")
    for field in fields:
        assert equal_nested(first[field], second[field]), f"Continuation changed {field}"
    assert metrics_without_time(Path(first_folder) / "metrics.csv") == metrics_without_time(
        Path(second_folder) / "metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(4)
    work = ROOT / "_verification" / ("extended_synthetic_" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True, exist_ok=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    report_file = ROOT / "reports" / "extended_verification.json"
    no_data = work / "no_official_data"
    started = time.monotonic()
    old_source = frozen_source_hashes()
    report = {"synthetic_verification_only": True, "formal_training_started": False,
              "official_test_partition_used": False, "device": args.device, "workers": args.workers,
              "artifact_directory": str(work), "checks": {}, "commands": [], "complete": False}

    def save_report():
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def command(script, arguments, expected_failure=False):
        call = [sys.executable, str(ROOT / script), *map(str, arguments)]
        result = subprocess.run(call, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300)
        number = len(report["commands"]) + 1
        (work / f"command_{number:02d}.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        report["commands"].append({"command": call, "returncode": result.returncode,
                                   "expected_failure": expected_failure})
        save_report()
        if expected_failure:
            assert result.returncode != 0, f"Unsafe/incomplete fixture accepted: {call}"
        else:
            assert result.returncode == 0, (result.stdout + result.stderr)[-7000:]

    def old_train(role, output, epochs=4, extra=()):
        command("train.py", ["--role", role, "--seed", 0, "--epochs", epochs,
                             "--data-root", no_data, "--output-root", output,
                             "--device", args.device, "--workers", args.workers,
                             "--synthetic", *extra])

    def train(case, output, epochs=2, extra=(), expected_failure=False):
        command("train_extended.py", ["--case-id", case, "--project-root", ROOT,
                 "--data-root", no_data, "--output-root", output, "--epochs", epochs,
                 "--device", args.device, "--workers", args.workers, "--synthetic", *extra], expected_failure)

    try:
        report["checks"].update(verify_protocol())
        report["checks"].update(verify_parameterized_objective())
        print("Verified extension matrix, shared defaults and parameterized KD gradients.", flush=True)

        original_full, original_partial = work / "legacy_full", work / "legacy_partial"
        inherited, full, resumed = work / "inherited", work / "extended_full", work / "extended_resumed"
        old_train("baseline", original_full)
        old_train("baseline", original_partial, extra=["--stop-after", 2])
        partial_folder = original_partial / "baseline_seed0"
        partial_before = tree_hashes(partial_folder)
        assert load_json(partial_folder / "summary.json")["completed_epochs"] == 2
        assert load_json(partial_folder / "summary.json")["status"] == "in_progress"
        train("baseline_seed0", inherited, epochs=4,
              extra=["--warm-start-checkpoint", partial_folder / "last.pth"])
        compare_boundaries(original_full / "baseline_seed0", inherited / "baseline_seed0")
        inherited_config = load_json(inherited / "baseline_seed0" / "config.json")
        assert inherited_config["warm_start"]["completed_epochs"] == 2
        assert inherited_config["warm_start"]["weights_sha256"] == partial_before["last.pth"]
        assert inherited_config["warm_start"]["original_config"] == load_json(partial_folder / "config.json")
        assert tree_hashes(partial_folder) == partial_before
        report["checks"]["legacy_partial_warm_start_exact_model_adam_rng_and_metrics"] = True
        report["checks"]["warm_start_preserves_original_checkpoint_config_and_log_bytes"] = True
        report["checks"]["warm_start_records_original_boundary_and_source_provenance"] = True
        print("Verified exact legacy-to-extension continuation without altering its source run.", flush=True)

        train("teacher_seed0", full)
        train("teacher_seed2", full)
        teacher_path = full / "teacher_seed0" / "best.pth"
        teacher_before = tree_hashes(teacher_path.parent)
        train("kd_seed0", full, epochs=4)
        common_teacher = ["--teacher-checkpoint", teacher_path]
        train("kd_seed0", resumed, epochs=4, extra=[*common_teacher, "--stop-after", 2])
        train("kd_seed0", resumed, epochs=4, extra=[*common_teacher, "--resume"])
        compare_boundaries(full / "kd_seed0", resumed / "kd_seed0", compare_config=True)
        train("kd_seed2", full)
        train("kd_T1_seed0", full)
        train("kd_A0p5_seed0", full)
        assert tree_hashes(teacher_path.parent) == teacher_before
        for case in ("kd_seed0", "kd_T1_seed0", "kd_A0p5_seed0"):
            config = load_json(full / case / "config.json")
            assert config["init_sha256"] == inherited_config["init_sha256"]
            assert config["teacher"]["weights_sha256"] == teacher_before["best.pth"]
            spec = protocol.case_spec(case)
            assert config["temperature"] == spec["temperature"] and config["alpha"] == spec["alpha"]
        assert load_json(full / "kd_seed2" / "config.json")["seed"] == 2
        report["checks"]["extended_kd_resume_exact_model_adam_rng_and_metrics"] = True
        report["checks"]["main_and_ablation_students_have_paired_initialization"] = True
        report["checks"]["temperature_and_alpha_cases_execute_declared_objective"] = True
        report["checks"]["third_seed_teacher_and_student_supported"] = True
        report["checks"]["teacher_artifact_bytes_unchanged_during_all_kd_cases"] = True

        from train_extended import load_teacher, validate_run
        expected = load_json(full / "kd_T1_seed0" / "config.json")
        teacher, _ = load_teacher(teacher_path, expected, torch.device(args.device))
        assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        before = model_state_sha256(teacher.state_dict())
        student = build_model("kd").to(args.device)
        images = torch.randn(4, 3, 32, 32, device=args.device)
        loss, _, _ = distillation_loss(student(images), teacher(images), torch.arange(4, device=args.device), 1.0, .25)
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        assert before == model_state_sha256(teacher.state_dict())
        del teacher, student, images, loss
        report["checks"]["nondefault_temperature_teacher_remains_eval_frozen"] = True

        for folder in (inherited / "baseline_seed0", full / "teacher_seed0", full / "kd_T1_seed0"):
            validate_run(folder / "best.pth", allow_synthetic=True)
            try:
                validate_run(folder / "best.pth")
            except ValueError:
                pass
            else:
                raise AssertionError("Synthetic run was accepted as formal")
        report["checks"]["formal_validator_rejects_synthetic_cases"] = True
        rejected = work / "rejected"
        train("kd_T4_seed0", rejected, extra=["--teacher-checkpoint", full / "teacher_seed2" / "best.pth"], expected_failure=True)
        train("kd_T4_seed0", rejected, extra=["--warm-start-checkpoint", partial_folder / "last.pth"], expected_failure=True)
        bad = work / "tampered" / "teacher_seed0"
        shutil.copytree(teacher_path.parent, bad)
        with (bad / "best.pth").open("ab") as stream:
            stream.write(b"synthetic_integrity_check")
        train("kd_T4_seed0", rejected, extra=["--teacher-checkpoint", bad / "best.pth"], expected_failure=True)
        assert not (rejected / "kd_T4_seed0").exists()
        report["checks"]["wrong_seed_and_tampered_teacher_rejected_before_run_creation"] = True
        report["checks"]["ablation_cannot_inherit_different_temperature_warm_training"] = True

        for folder in (inherited / "baseline_seed0", full / "kd_T1_seed0", full / "kd_A0p5_seed0"):
            command("evaluate_extended.py", ["--project-root", ROOT, "--runs-root", folder.parent,
                    "--output-root", work / "nonformal_evaluation", "--checkpoint", folder / "best.pth",
                    "--data-root", no_data, "--device", args.device, "--workers", args.workers,
                    "--synthetic", "--allow-nonformal", "--latency-warmup", 1, "--latency-repeats", 2])
            evaluated = load_json(folder / "evaluation.json")
            assert evaluated["formal"] is False and evaluated["total"] == 64
            assert evaluated["partition"] == "synthetic_validation_for_verification"
        report["checks"]["synthetic_extended_evaluations_explicitly_nonformal"] = True
        command("summarize_extended.py", ["--project-root", ROOT, "--runs-root", full,
                "--output-root", work / "rejected_results"], expected_failure=True)
        assert not list((work / "rejected_results").glob("*.csv"))
        report["checks"]["formal_summary_rejects_incomplete_synthetic_matrix"] = True
        assert not no_data.exists(), "Synthetic verification unexpectedly accessed real-data storage"
        assert frozen_source_hashes() == old_source, "Completed study source was modified"
        assert tree_hashes(partial_folder) == partial_before
        assert tree_hashes(teacher_path.parent) == teacher_before
        report["checks"]["original_source_vendor_and_protocol_bytes_preserved"] = True
        report["complete"] = True
        report["all_checks_passed"] = all(report["checks"].values())
        report["elapsed_seconds"] = time.monotonic() - started
        save_report()
        print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2), flush=True)
    except BaseException as error:
        report["all_checks_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.monotonic() - started
        save_report()
        raise


if __name__ == "__main__":
    main()
