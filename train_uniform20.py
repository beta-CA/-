"""Train an independent case with a uniform 20-epoch teacher/student budget.

Derived from the project's verified train_extended.py. Model, objective, data,
RNG handling, boundary auditing and atomic file helpers are reused. Historical
training scripts and results remain unchanged; no historical warm start is used.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import uniform_protocol as protocol
import train as legacy
from train_extended import read_boundary, read_log, require
from models import build_model, distillation_loss
from kd_utils import (MEAN, STD, best_payload, capture_rng, evaluate_loader,
                      file_sha256, load_checkpoint, load_json, make_datasets,
                      model_state_sha256, restore_rng, save_checkpoint, save_json,
                      seed_everything, seed_worker, software_information,
                      validate_csv, verify_best, write_csv)


def validate_case_config(config, allow_synthetic=False):
    require(config.get("study") == protocol.STUDY, "Not a uniform-20 checkpoint")
    spec = protocol.case_spec(config.get("case_id"))
    synthetic = config.get("dataset") == "synthetic"
    require(config.get("dataset") in ("CIFAR10", "synthetic"), "Unexpected data identity")
    require(allow_synthetic or not synthetic, "Synthetic verification is not a formal experiment")
    require(config.get("verification_only") is synthetic, "Verification/formal identity is inconsistent")
    require(config.get("warm_start") is None, "Uniform-20 cases cannot inherit old training")
    require(type(config.get("epochs")) is int and config["epochs"] >= 1, "Invalid epoch budget")
    for key, value in (("role", spec["role"]), ("seed", spec["seed"]), ("group", spec["group"]),
                       ("temperature", spec["temperature"]), ("alpha", spec["alpha"]),
                       ("optimizer", legacy.OPTIMIZER), ("normalize_mean", MEAN),
                       ("normalize_std", STD), ("precision", "float32"), ("augmentation", "none")):
        require(config.get(key) == value, f"Case configuration differs: {key}")
    project = Path(config["project_root"]).resolve(strict=True)
    output = protocol.validate_output_root(project, config["output_root"])
    require(str(output) == config["output_root"], "Output root metadata is not canonical")
    if not synthetic:
        require(config["epochs"] == protocol.TRAINING_EPOCHS and config.get("batch_size") == 128
                and config.get("workers") == 2 and config.get("device") in ("cuda", "cuda:0")
                and config.get("deterministic") is True and config.get("tf32") is False
                and config.get("software", {}).get("cpu_threads") == 4
                and config.get("split", {}).get("split_seed") == 20261026
                and config.get("split", {}).get("train_size") == 45000
                and config.get("split", {}).get("val_size") == 5000,
                "Not the fixed formal 20-epoch configuration")
    return spec, synthetic


def validate_run(checkpoint_path, allow_synthetic=False):
    """Return (best,last,summary,best-file-sha) after read-only boundary audits."""
    path = Path(checkpoint_path).resolve(strict=True)
    require(path.name == "best.pth", "Use validation-selected best.pth")
    best, last, summary, _, digest, _ = read_boundary(path.parent, protocol.training_source())
    spec, synthetic = validate_case_config(best["config"], allow_synthetic)
    config = best["config"]
    require(path.parent.name == spec["run_id"], "Case directory identity differs")
    require(path.parent.parent == Path(config["output_root"]).resolve(), "Run is outside its declared output root")
    if config["role"] == "kd":
        dependency = config.get("teacher")
        require(isinstance(dependency, dict), "KD teacher dependency is missing")
        teacher_path = Path(dependency["checkpoint"]).resolve(strict=True)
        teacher_best, _, teacher_hash = validate_teacher(teacher_path, config)
        expected_dependency = {"checkpoint": str(teacher_path), "weights_sha256": teacher_hash,
                               "model_state_sha256": teacher_best["model_state_sha256"], "seed": config["seed"]}
        require(dependency == expected_dependency, "KD teacher dependency metadata changed")
    else:
        require(config.get("teacher") is None, "Teacher/baseline must not depend on a teacher")
    return best, last, summary, digest


def validate_teacher(path, expected):
    """Accept only a completed same-study/same-root/same-seed teacher."""
    path = Path(path).resolve(strict=True)
    declared = protocol.teacher_checkpoint(expected["project_root"], expected["output_root"], expected["seed"])
    require(path == declared.resolve(), "KD cannot use a historical or external teacher checkpoint")
    config = load_json(path.parent / "config.json")
    require(config.get("study") == protocol.STUDY, "Historical teacher is not a uniform-20 teacher")
    require(config.get("role") == "teacher" and config.get("teacher") is None,
            "Dependency is not an independently trained teacher")
    synthetic = expected["dataset"] == "synthetic"
    best, _, summary, digest = validate_run(path, allow_synthetic=synthetic)
    require(config.get("epochs") == expected["epochs"] == summary.get("completed_epochs")
            == summary.get("requested_epochs"), "Teacher and student must share the complete training budget")
    require(config.get("temperature") == protocol.DEFAULT_TEMPERATURE
            and config.get("alpha") == protocol.DEFAULT_ALPHA, "Teacher recipe differs from the fixed default")
    for key in ("study", "seed", "batch_size", "workers", "device", "precision", "dataset", "data_root",
                "project_root", "output_root", "split_sha256", "optimizer", "normalize_mean", "normalize_std",
                "software", "source", "verification_only"):
        require(config.get(key) == expected[key], f"Teacher and student differ: {key}")
    return best, summary, digest


def load_teacher(path, expected, device):
    path = Path(path).resolve(strict=True)
    best, _, digest = validate_teacher(path, expected)
    model = build_model("teacher").float().to(device)
    model.load_state_dict(best["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model, {"checkpoint": str(path), "weights_sha256": digest,
                   "model_state_sha256": best["model_state_sha256"], "seed": best["config"]["seed"]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Tagged tiny verification data, never a formal result")
    parser.add_argument("--epochs", type=int, help="Synthetic verification only; all formal cases use 20")
    parser.add_argument("--stop-after", type=int, help="Stop at a committed epoch without changing the budget")
    args = parser.parse_args()
    spec = protocol.case_spec(args.case_id)
    if args.epochs is not None and not args.synthetic:
        parser.error("All formal cases use twenty epochs; --epochs is synthetic-only")
    args.epochs = args.epochs if args.epochs is not None else protocol.TRAINING_EPOCHS
    if args.epochs < 1 or args.workers < 0 or (args.stop_after is not None and not 0 <= args.stop_after <= args.epochs):
        parser.error("Invalid epoch/workers/stop-after budget")
    if spec["role"] != "kd" and args.teacher_checkpoint is not None:
        parser.error("Only KD has a teacher dependency")
    return args, spec


def main():
    args, spec = parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    require(device.type in ("cpu", "cuda"), "Only CPU and CUDA are supported")
    require(device.type != "cuda" or torch.cuda.is_available(), "Requested CUDA is unavailable")
    project = args.project_root.resolve(strict=True)
    output = protocol.validate_output_root(project, args.output_root or project / protocol.RUNS_DIRECTORY)
    data_root = (args.data_root or project / "data").resolve()
    require(not output.is_relative_to(data_root) and not data_root.is_relative_to(output),
            "Output and dataset directories must not overlap")
    if not args.synthetic:
        protocol.verify_manifest(project)
        require(args.workers == 2 and str(device) in ("cuda", "cuda:0"), "Formal training uses CUDA/workers2")
    run_dir = output / spec["run_id"]
    require(not args.resume or run_dir.is_dir(), f"No uniform-20 case to resume: {run_dir}")
    require(args.resume or not run_dir.exists(), f"Refusing to overwrite {run_dir}; use --resume")
    training, validation, split = make_datasets(data_root, output / "splits", args.synthetic)
    expected = {"schema_version": 1, "study": protocol.STUDY, "case_id": spec["run_id"],
                "group": spec["group"], "role": spec["role"], "seed": spec["seed"],
                "dataset": "synthetic" if args.synthetic else "CIFAR10",
                "source_partition": "synthetic" if args.synthetic else "train", "epochs": args.epochs,
                "verification_only": bool(args.synthetic), "batch_size": 128, "workers": args.workers,
                "device": str(device), "precision": "float32", "data_root": str(data_root),
                "project_root": str(project), "output_root": str(output),
                "split_seed": split["split_seed"], "split_sha256": split["split_sha256"],
                "split": {key: split[key] for key in ("split_seed", "train_size", "val_size", "split_sha256")},
                "normalize_mean": MEAN, "normalize_std": STD, "augmentation": "none",
                "optimizer": legacy.OPTIMIZER, "temperature": spec["temperature"], "alpha": spec["alpha"],
                "teacher": None, "warm_start": None, "deterministic": True, "tf32": False,
                "accuracy_unit": "fraction", "software": software_information(device),
                "source": protocol.training_source()}
    teacher = None
    if spec["role"] == "kd":
        teacher_path = args.teacher_checkpoint or protocol.teacher_checkpoint(project, output, spec["seed"])
        teacher, expected["teacher"] = load_teacher(teacher_path, expected, device)
    # Load the teacher BEFORE resetting the seed, preserving paired student RNG.
    seed_everything(spec["seed"])
    model = build_model(spec["role"])
    init_hash = model_state_sha256(model.state_dict())
    config = {**expected, "init_sha256": init_hash}
    validate_case_config(config, allow_synthetic=args.synthetic)
    train_generator = torch.Generator().manual_seed(spec["seed"])
    val_generator = torch.Generator().manual_seed(spec["seed"] + 10000)
    options = dict(batch_size=128, num_workers=args.workers, pin_memory=device.type == "cuda",
                   worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(training, shuffle=True, generator=train_generator, **options)
    val_loader = DataLoader(validation, shuffle=False, generator=val_generator, **options)
    model = model.float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    if args.resume:
        require(load_json(run_dir / "config.json") == config, "Resume configuration or teacher dependency changed")
        checkpoint = load_checkpoint(run_dir / "last.pth")
        require(checkpoint.get("config") == config and checkpoint.get("init_sha256") == init_hash,
                "Last checkpoint configuration differs")
        require(0 <= checkpoint["epoch"] <= args.epochs, "Invalid saved epoch")
        require(model_state_sha256(checkpoint["model_state"]) == checkpoint["model_state_sha256"],
                "Last model tensor hash failed")
        rows = validate_csv(run_dir / "metrics.csv", checkpoint)
        verify_best(run_dir / "best.pth", checkpoint)
    else:
        output.mkdir(parents=True, exist_ok=True)
        initial = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.initializing-", dir=output))
        rows = []
        checkpoint = {"schema_version": 1, "config": config, "epoch": 0,
                      "model_state": model.state_dict(), "model_state_sha256": init_hash,
                      "init_sha256": init_hash, "optimizer_state": optimizer.state_dict(),
                      "rng_state": capture_rng(train_generator, val_generator), "best_epoch": 0,
                      "best_val_acc": -1.0, "best_val_loss": None,
                      "best_model_state_sha256": None, "last_metrics": None}
        save_json(initial / "config.json", config)
        write_csv(initial / "metrics.csv", rows)
        save_checkpoint(initial / "last.pth", checkpoint)
        legacy.write_summary(initial, checkpoint, args.epochs)
        require(not run_dir.exists(), f"Another process created {run_dir}; initialization retained at {initial}")
        initial.rename(run_dir)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_rng(checkpoint["rng_state"], train_generator, val_generator)
    legacy.write_summary(run_dir, checkpoint, args.epochs)
    print(f"Uniform-20 case: {spec['run_id']}; completed={checkpoint['epoch']}/{args.epochs}; "
          f"T={spec['temperature']}; alpha={spec['alpha']}; verification_only={args.synthetic}", flush=True)
    target = min(args.epochs, args.stop_after) if args.stop_after is not None else args.epochs
    for epoch in range(checkpoint["epoch"] + 1, target + 1):
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        if teacher is not None: teacher.eval()
        sums, correct, count = {"total": 0.0, "ce": 0.0, "kd": 0.0}, 0, 0
        batches = tqdm(train_loader, desc=f"{spec['run_id']} epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for images, labels in batches:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if teacher is None:
                ce = nn.functional.cross_entropy(logits, labels)
                kd, loss = ce.new_zeros(()), ce
            else:
                with torch.no_grad(): targets = teacher(images)
                loss, ce, kd = distillation_loss(logits, targets, labels, spec["temperature"], spec["alpha"])
            require(bool(torch.isfinite(loss)), "Non-finite loss; previous epoch remains committed")
            loss.backward()
            optimizer.step()
            size = labels.numel()
            for key, value in (("total", loss), ("ce", ce), ("kd", kd)):
                sums[key] += value.detach().item() * size
            correct += int((logits.detach().argmax(1) == labels).sum().item())
            count += size
            batches.set_postfix(total_loss=f"{sums['total']/count:.4f}", train_acc=f"{correct/count:.4f}")
        require(count == len(training), "Training did not cover every sample")
        result = evaluate_loader(model, val_loader, device)
        require(result["count"] == len(validation), "Validation did not cover every sample")
        if device.type == "cuda": torch.cuda.synchronize(device)
        row = {"epoch": epoch, "train_total_loss": sums["total"]/count, "train_ce_loss": sums["ce"]/count,
               "train_kd_loss": sums["kd"]/count, "train_acc": correct/count, "val_loss": result["loss"],
               "val_acc": result["accuracy"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter()-started}
        require(all(np.isfinite(v) for v in row.values()), "Non-finite epoch metrics")
        improved = row["val_acc"] > checkpoint["best_val_acc"]
        state_hash = model_state_sha256(model.state_dict())
        checkpoint = {"schema_version": 1, "config": config, "epoch": epoch, "model_state": model.state_dict(),
                      "model_state_sha256": state_hash, "init_sha256": init_hash,
                      "optimizer_state": optimizer.state_dict(), "rng_state": capture_rng(train_generator, val_generator),
                      "best_epoch": epoch if improved else checkpoint["best_epoch"],
                      "best_val_acc": row["val_acc"] if improved else checkpoint["best_val_acc"],
                      "best_val_loss": row["val_loss"] if improved else checkpoint["best_val_loss"],
                      "best_model_state_sha256": state_hash if improved else checkpoint["best_model_state_sha256"],
                      "last_metrics": row}
        save_checkpoint(run_dir / "last.pth", checkpoint)
        if improved: save_checkpoint(run_dir / "best.pth", best_payload(checkpoint))
        rows.append(row)
        write_csv(run_dir / "metrics.csv", rows)
        legacy.write_summary(run_dir, checkpoint, args.epochs)
        print(f"epoch={epoch} val_acc={row['val_acc']:.6f} best_epoch={checkpoint['best_epoch']}", flush=True)
    print(f"Committed {checkpoint['epoch']}/{args.epochs}: {run_dir}", flush=True)
    return checkpoint


if __name__ == "__main__":
    main()
