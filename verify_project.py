"""Verify the implementation using synthetic fixtures only, never paper results.

No official test partition or formal runs/ directory is touched. Artifacts are
kept in a uniquely named _verification/ directory for inspection. No claim of
classification performance may be made from this verification.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ast
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

from models import build_model, distillation_loss, source_information
from kd_utils import ensure_split, file_sha256, load_checkpoint, load_json, model_state_sha256

ROOT = Path(__file__).resolve().parent


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def read_metrics(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{key: value for key, value in row.items() if key != "seconds"}
                for row in csv.DictReader(stream)]


def verify_models_and_objective():
    source_information()  # Validates all upstream file hashes, including license.
    reference = ast.parse((ROOT / "vendor/pytorch_tutorial/reference_source.txt").read_text(encoding="utf-8"))
    extracted = ast.parse((ROOT / "vendor/pytorch_tutorial/models.py").read_text(encoding="utf-8"))
    for name in ("DeepNN", "LightNN"):
        first = next(node for node in reference.body if isinstance(node, ast.ClassDef) and node.name == name)
        second = next(node for node in extracted.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.dump(first, include_attributes=False) == ast.dump(second, include_attributes=False)
    assert sum(p.numel() for p in build_model("teacher").parameters()) == 1_186_986
    assert sum(p.numel() for p in build_model("baseline").parameters()) == 267_738
    torch.manual_seed(91)
    baseline = build_model("baseline")
    torch.manual_seed(91)
    student = build_model("kd")
    assert model_state_sha256(baseline.state_dict()) == model_state_sha256(student.state_dict())

    generator = torch.Generator().manual_seed(123)
    initial = torch.randn(7, 10, dtype=torch.float64, generator=generator)
    teacher = torch.randn(7, 10, dtype=torch.float64, generator=generator, requires_grad=True)
    labels = torch.arange(7) % 10
    student_logits = initial.clone().requires_grad_()
    reference_logits = initial.clone().requires_grad_()
    total, ce, kd = distillation_loss(student_logits, teacher, labels, temperature=2.0, alpha=0.25)
    targets = F.softmax(teacher.detach() / 2.0, dim=-1)
    log_student = F.log_softmax(reference_logits / 2.0, dim=-1)
    reference_kd = torch.sum(targets * (targets.log() - log_student)) / 7 * 4.0
    reference_ce = F.cross_entropy(reference_logits, labels)
    expected = 0.75 * reference_ce + 0.25 * reference_kd
    torch.testing.assert_close(total, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ce, reference_ce, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(kd, reference_kd, rtol=1e-12, atol=1e-12)
    total.backward()
    expected.backward()
    torch.testing.assert_close(student_logits.grad, reference_logits.grad, rtol=1e-12, atol=1e-12)
    assert teacher.grad is None
    for alpha in (0.0, 1.0):
        value, hard, soft = distillation_loss(initial, teacher, labels, alpha=alpha)
        torch.testing.assert_close(value, hard if alpha == 0.0 else soft)
    extreme = torch.tensor([[1e4, -1e4, 0.0] * 3 + [1.0]], requires_grad=True)
    value, _, _ = distillation_loss(extreme, -extreme.detach(), torch.tensor([1]))
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(extreme.grad).all()
    for temperature, alpha in ((0.0, .25), (float("inf"), .25), (float("nan"), .25), (2., -.1), (2., 1.1)):
        try:
            distillation_loss(initial, teacher, labels, temperature=temperature, alpha=alpha)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid KD hyperparameter was accepted")
    return {"model_classes_match_official_ast": True,
            "teacher_parameters": 1_186_986, "student_parameters": 267_738,
            "paired_initialization": True, "tutorial_loss_and_gradient_equivalence": True,
            "teacher_gradient_detached": True, "extreme_logits_finite": True,
            "invalid_hyperparameters_rejected": True}


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
    work = ROOT / "_verification" / ("synthetic_" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True, exist_ok=False)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "verification.json"
    started = time.monotonic()
    report = {"synthetic_verification_only": True, "formal_training_started": False,
              "official_test_partition_used": False, "device": args.device,
              "workers": args.workers, "artifact_directory": str(work), "checks": {},
              "commands": [], "complete": False}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def command(script, arguments, expected_failure=False, expected_message=None):
        call = [sys.executable, str(ROOT / script), *map(str, arguments)]
        result = subprocess.run(call, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=240)
        number = len(report["commands"]) + 1
        (work / f"command_{number:02d}.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        report["commands"].append({"command": call, "returncode": result.returncode,
                                   "expected_failure": expected_failure})
        save_report()
        output = result.stdout + result.stderr
        if expected_failure:
            assert result.returncode != 0, f"Unsafe input was accepted: {call}"
            if expected_message:
                assert expected_message.lower() in output.lower(), output[-4000:]
        else:
            assert result.returncode == 0, output[-6000:]

    runs = work / "runs"
    resumed = work / "resumed"
    # A deliberately nonexistent data directory proves synthetic training does not
    # need any CIFAR files. Nothing in this command opens the official test split.
    no_data = work / "no_official_data"

    def train(role, seed, output=runs, extra=(), expected_failure=False, expected_message=None):
        arguments = ["--role", role, "--seed", seed, "--epochs", 2, "--batch-size", 128,
                     "--workers", args.workers, "--device", args.device,
                     "--data-root", no_data, "--output-root", output, "--synthetic", *extra]
        command("train.py", arguments, expected_failure, expected_message)

    try:
        report["checks"].update(verify_models_and_objective())
        print("Verified upstream models and distillation objective/gradients.", flush=True)
        # This fixture exercises the real splitter without loading any images.
        fixture_labels = np.repeat(np.arange(10), 5000)
        split_folder = work / "split_index_fixture"
        split = ensure_split(split_folder, fixture_labels)
        train_indices, val_indices = split["train_indices"], split["val_indices"]
        assert len(train_indices) == 45000 and len(val_indices) == 5000
        assert set(train_indices).isdisjoint(val_indices)
        assert sorted(train_indices + val_indices) == list(range(50000))
        assert np.array_equal(np.bincount(fixture_labels[val_indices]), np.full(10, 500))
        assert ensure_split(split_folder, fixture_labels) == split
        split_path = next(split_folder.glob("split_cifar10_seed*.json"))
        original_split = split_path.read_bytes()
        tampered_split = dict(split)
        tampered_split["val_indices"] = [0] * 5000
        split_path.write_text(json.dumps(tampered_split), encoding="utf-8")
        try:
            ensure_split(split_folder, fixture_labels)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered split indices were accepted")
        split_path.write_bytes(original_split)
        report["checks"]["stratified_split_disjoint_complete_and_tamper_checked"] = True
        for seed in (0, 1):
            train("teacher", seed)
            train("baseline", seed)
            teacher_path = runs / f"teacher_seed{seed}" / "best.pth"
            teacher_hash = file_sha256(teacher_path)
            train("kd", seed)
            assert file_sha256(teacher_path) == teacher_hash
            baseline_config = load_json(runs / f"baseline_seed{seed}" / "config.json")
            kd_config = load_json(runs / f"kd_seed{seed}" / "config.json")
            assert baseline_config["init_sha256"] == kd_config["init_sha256"]
            assert kd_config["teacher"]["weights_sha256"] == teacher_hash
            assert kd_config["teacher"]["seed"] == seed
            assert load_json(runs / f"kd_seed{seed}" / "summary.json")["dataset"] == "synthetic"
            print(f"Verified synthetic teacher/baseline/KD pair seed {seed}.", flush=True)
        report["checks"]["both_synthetic_seed_pairs_complete"] = True
        report["checks"]["teacher_checkpoint_unchanged_during_kd"] = True
        report["checks"]["trained_student_pair_initialization_matches"] = True

        # Inspect and exercise the exact loader used by the trainer: eval mode,
        # disabled parameter gradients, and no teacher gradients after KD backward.
        from train import load_teacher
        expected = load_json(runs / "kd_seed0" / "config.json")
        teacher, _ = load_teacher(runs / "teacher_seed0" / "best.pth", expected, torch.device(args.device))
        assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        student = build_model("kd").to(args.device)
        images = torch.randn(4, 3, 32, 32, device=args.device)
        teacher_state = model_state_sha256(teacher.state_dict())
        value, _, _ = distillation_loss(student(images), teacher(images), torch.arange(4, device=args.device))
        value.backward()
        assert all(p.grad is None for p in teacher.parameters())
        assert teacher_state == model_state_sha256(teacher.state_dict())
        del teacher, student, images, value
        report["checks"]["trainer_teacher_eval_and_frozen_backward"] = True

        train("baseline", 0, resumed, ["--stop-after", 1])
        paused = resumed / "baseline_seed0"
        assert load_json(paused / "summary.json")["completed_epochs"] == 1
        # Simulate derived artifacts not having committed after last.pth. Only
        # files created by this synthetic verification are modified here.
        (paused / "best.pth").unlink()
        header = (paused / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
        (paused / "metrics.csv").write_text(header + "\n", encoding="utf-8")
        (paused / "summary.json").unlink()
        train("baseline", 0, resumed, ["--resume"])
        full = load_checkpoint(runs / "baseline_seed0" / "last.pth")
        recovered = load_checkpoint(paused / "last.pth")
        for field in ("config", "model_state", "optimizer_state", "rng_state", "init_sha256",
                      "best_epoch", "best_val_acc", "best_model_state_sha256"):
            assert equal_nested(full[field], recovered[field]), f"Recovery changed {field}"
        assert read_metrics(runs / "baseline_seed0" / "metrics.csv") == read_metrics(paused / "metrics.csv")
        report["checks"]["epoch_boundary_resume_exact_model_optimizer_rng_metrics"] = True
        report["checks"]["best_csv_summary_recovered_after_interrupted_commit"] = True
        print("Verified exact epoch-boundary recovery and interrupted derived-file repair.", flush=True)

        # Invalid teacher dependencies must fail before creating the KD run.
        rejected = work / "rejected"
        train("kd", 1, rejected, ["--teacher-checkpoint", runs / "teacher_seed0" / "best.pth"],
              expected_failure=True, expected_message="seed")
        assert not (rejected / "kd_seed1").exists()
        bad = work / "tampered_teacher" / "teacher_seed0"
        shutil.copytree(runs / "teacher_seed0", bad)
        with (bad / "best.pth").open("ab") as stream:
            stream.write(b"verification_corruption")
        train("kd", 0, rejected, ["--teacher-checkpoint", bad / "best.pth"],
              expected_failure=True, expected_message="best.pth differs")
        assert not (rejected / "kd_seed0").exists()
        report["checks"]["wrong_seed_teacher_rejected_before_run_creation"] = True
        report["checks"]["tampered_teacher_rejected_before_run_creation"] = True

        for seed in (0, 1):
            for role in ("teacher", "baseline", "kd"):
                arguments = ["--checkpoint", runs / f"{role}_seed{seed}" / "best.pth",
                             "--data-root", no_data, "--device", args.device, "--workers", args.workers,
                             "--synthetic", "--allow-nonformal", "--latency-warmup", 1,
                             "--latency-repeats", 2]
                command("evaluate.py", arguments)
                evaluation = load_json(runs / f"{role}_seed{seed}" / "evaluation.json")
                assert evaluation["formal"] is False and evaluation["total"] == 64
                assert evaluation["partition"] == "synthetic_validation_for_verification"
        report["checks"]["synthetic_evaluation_explicitly_nonformal"] = True
        command("summarize.py", ["--runs-root", runs, "--output-root", work / "rejected_synthetic_results"],
                expected_failure=True, expected_message="synthetic")
        empty = work / "empty_runs"
        empty.mkdir()
        command("summarize.py", ["--runs-root", empty, "--output-root", work / "rejected_missing_results"],
                expected_failure=True, expected_message="exactly six")
        assert not list((work / "rejected_synthetic_results").glob("*.csv"))
        assert not list((work / "rejected_missing_results").glob("*.csv"))
        assert not no_data.exists(), "A synthetic test unexpectedly accessed/created real-data storage"
        report["checks"]["summarizer_rejects_complete_synthetic_matrix"] = True
        report["checks"]["summarizer_rejects_missing_matrix"] = True
        report["complete"] = True
        report["all_checks_passed"] = all(report["checks"].values())
        report["elapsed_seconds"] = time.monotonic() - started
        save_report()
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["all_checks_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.monotonic() - started
        save_report()
        raise


if __name__ == "__main__":
    main()
