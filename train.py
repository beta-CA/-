"""Train one teacher, baseline student, or distillation student.

Networks and KD objective follow the PyTorch official knowledge-distillation
tutorial. Checkpoint/RNG/atomic I/O helpers are reused from the earlier Cutout
project. This entry point uses only the official TRAIN partition; model selection
uses a fixed held-out validation partition. All accuracies here are fractions.
Resume restores the last committed epoch and reruns any interrupted epoch.
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

from models import build_model, distillation_loss, source_information
from kd_utils import (MEAN, STD, SPLIT_SEED, capture_rng, evaluate_loader, file_sha256,
                      load_checkpoint, load_json, make_datasets, model_state_sha256,
                      restore_rng, save_checkpoint, save_json, seed_everything,
                      seed_worker, software_information, validate_csv, verify_best,
                      write_csv, best_payload)


OPTIMIZER = {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999],
             "eps": 1e-8, "weight_decay": 0.0}
TEMPERATURE = 2.0
ALPHA = 0.25


def training_source():
    root = Path(__file__).resolve().parent
    return {"models": source_information(),
            "train_py_sha256": file_sha256(root / "train.py"),
            "kd_utils_py_sha256": file_sha256(root / "kd_utils.py")}


def write_summary(run_dir, checkpoint, epochs):
    best_epoch = checkpoint["best_epoch"]
    config = checkpoint["config"]
    save_json(run_dir / "summary.json", {
        "schema_version": 1,
        "status": "completed" if checkpoint["epoch"] == epochs else "in_progress",
        "dataset": config["dataset"], "role": config["role"], "seed": config["seed"],
        "completed_epochs": checkpoint["epoch"], "requested_epochs": epochs,
        "best_epoch": best_epoch or None,
        "best_val_acc": checkpoint["best_val_acc"] if best_epoch else None,
        "best_val_loss": checkpoint["best_val_loss"], "accuracy_unit": "fraction",
        "selection": "highest validation accuracy; earliest epoch wins ties",
        "init_sha256": checkpoint["init_sha256"],
        "split_sha256": config["split_sha256"],
        "best_model_state_sha256": checkpoint["best_model_state_sha256"],
        "best_checkpoint_sha256": file_sha256(run_dir / "best.pth") if best_epoch else None,
        "last_checkpoint_sha256": file_sha256(run_dir / "last.pth"),
        "resume_policy": "epoch boundary only; interrupted epoch is rerun; same environment required",
        "train_metric_note": "training batches are measured during optimization, not a fixed clean train evaluation",
        "test_evaluated": False,
    })


def load_teacher(path, expected, device):
    """Require the completed same-seed teacher's validation-selected checkpoint."""
    path = Path(path).resolve(strict=True)
    if path.name != "best.pth":
        raise ValueError("KD must use the teacher's validation-selected best.pth")
    folder = path.parent
    summary = load_json(folder / "summary.json")
    config = load_json(folder / "config.json")
    if (summary.get("status") != "completed" or
            summary.get("completed_epochs") != expected["epochs"] or
            summary.get("requested_epochs") != expected["epochs"]):
        raise ValueError("The teacher must complete its requested training before KD starts")
    if config.get("role") != "teacher" or config.get("teacher") is not None:
        raise ValueError("Teacher checkpoint is not an independently trained teacher")
    common_fields = ("seed", "epochs", "batch_size", "workers", "device", "precision",
                     "dataset", "data_root", "split_sha256", "optimizer", "temperature",
                     "alpha", "normalize_mean", "normalize_std", "software", "source")
    for key in common_fields:
        if config.get(key) != expected[key]:
            raise ValueError(f"Teacher and student differ in required field: {key}")
    digest = file_sha256(path)
    if summary.get("best_checkpoint_sha256") != digest:
        raise ValueError("Teacher best.pth differs from its completed summary")
    checkpoint = load_checkpoint(path)
    state_hash = model_state_sha256(checkpoint["model_state"])
    if (checkpoint.get("config") != config or checkpoint.get("epoch") != summary.get("best_epoch") or
            checkpoint.get("best_epoch") != summary.get("best_epoch") or
            checkpoint.get("model_state_sha256") != state_hash or
            checkpoint.get("best_model_state_sha256") != state_hash or
            summary.get("best_model_state_sha256") != state_hash or
            checkpoint.get("best_val_acc") != summary.get("best_val_acc")):
        raise ValueError("Teacher best checkpoint selection/configuration/content is inconsistent")
    last_path = folder / "last.pth"
    if file_sha256(last_path) != summary.get("last_checkpoint_sha256"):
        raise ValueError("Teacher last.pth differs from its completed summary")
    last = load_checkpoint(last_path)
    if (last.get("config") != config or last.get("epoch") != expected["epochs"] or
            last.get("best_epoch") != checkpoint["epoch"] or
            last.get("best_model_state_sha256") != state_hash):
        raise ValueError("Teacher completed epoch boundary is inconsistent")
    model = build_model("teacher").float().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if file_sha256(path) != digest:
        raise RuntimeError("Teacher checkpoint changed while loading")
    return model, {"checkpoint": str(path), "weights_sha256": digest,
                   "model_state_sha256": state_hash, "seed": config["seed"]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("teacher", "baseline", "kd"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu; no silent CPU fallback")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use tagged tiny synthetic data for verification only")
    parser.add_argument("--stop-after", type=int, help="Stop at this completed epoch boundary; keeps --epochs unchanged")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    if args.stop_after is not None and not 0 <= args.stop_after <= args.epochs:
        parser.error("stop-after must be between 0 and requested epochs")
    if args.role != "kd" and args.teacher_checkpoint is not None:
        parser.error("Only KD uses --teacher-checkpoint")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; select the configured CUDA environment")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device does not exist")
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    run_dir = output_root / f"{args.role}_seed{args.seed}"
    if args.resume and not run_dir.is_dir():
        raise FileNotFoundError(f"No run to resume: {run_dir}")
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}; use --resume for the identical configuration")
    training_data, validation_data, split = make_datasets(data_root, output_root / "splits", args.synthetic)
    expected = {
        "schema_version": 1, "role": args.role, "seed": args.seed,
        "dataset": "synthetic" if args.synthetic else "CIFAR10", "source_partition": "train" if not args.synthetic else "synthetic",
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "device": str(device), "precision": "float32", "data_root": str(data_root),
        "split_seed": split["split_seed"], "split_sha256": split["split_sha256"],
        "split": {key: split[key] for key in ("split_seed", "train_size", "val_size", "split_sha256")},
        "normalize_mean": MEAN, "normalize_std": STD, "augmentation": "none",
        "optimizer": OPTIMIZER, "temperature": TEMPERATURE, "alpha": ALPHA,
        "teacher": None, "deterministic": True, "tf32": False,
        "accuracy_unit": "fraction", "software": software_information(device),
        "source": training_source(),
    }
    teacher = None
    if args.role == "kd":
        teacher_path = args.teacher_checkpoint or output_root / f"teacher_seed{args.seed}" / "best.pth"
        teacher, expected["teacher"] = load_teacher(teacher_path, expected, device)
    # Teacher construction/loading must happen BEFORE this seed reset, so the
    # student pair starts with identical weights, shuffle order and dropout RNG.
    seed_everything(args.seed)
    model = build_model(args.role)
    init_hash = model_state_sha256(model.state_dict())
    config = {**expected, "init_sha256": init_hash}
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 10000)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                          persistent_workers=False)
    train_loader = DataLoader(training_data, shuffle=True, generator=train_generator, **loader_options)
    val_loader = DataLoader(validation_data, shuffle=False, generator=val_generator, **loader_options)
    model = model.float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    if args.resume:
        if load_json(run_dir / "config.json") != config:
            raise ValueError("Resume configuration changed (data, teacher, source, runtime, seed or optimizer)")
        checkpoint = load_checkpoint(run_dir / "last.pth")
        if checkpoint.get("config") != config or checkpoint.get("init_sha256") != init_hash:
            raise ValueError("last.pth configuration/initialization differs")
        if not 0 <= checkpoint["epoch"] <= args.epochs:
            raise ValueError("last.pth epoch is outside the requested training budget")
        if model_state_sha256(checkpoint["model_state"]) != checkpoint["model_state_sha256"]:
            raise ValueError("last.pth model content hash failed")
        rows = validate_csv(run_dir / "metrics.csv", checkpoint)
        verify_best(run_dir / "best.pth", checkpoint)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng_state"], train_generator, val_generator)
        print(f"Resuming after completed epoch {checkpoint['epoch']}; interrupted epoch is rerun.", flush=True)
    else:
        # Publish epoch0 only after config, CSV, checkpoint and summary all exist.
        # A power loss during initialization leaves an unadvertised diagnostic
        # folder instead of a formal run directory with no recoverable boundary.
        output_root.mkdir(parents=True, exist_ok=True)
        initial_dir = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.initializing-", dir=output_root))
        if initial_dir.parent.resolve() != output_root or run_dir.parent.resolve() != output_root:
            raise RuntimeError("Initialization must stay inside the requested output root")
        save_json(initial_dir / "config.json", config)
        rows = []
        write_csv(initial_dir / "metrics.csv", rows)
        checkpoint = {
            "schema_version": 1, "config": config, "epoch": 0,
            "model_state": model.state_dict(), "model_state_sha256": init_hash,
            "init_sha256": init_hash, "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng(train_generator, val_generator),
            "best_epoch": 0, "best_val_acc": -1.0, "best_val_loss": None,
            "best_model_state_sha256": None, "last_metrics": None,
        }
        save_checkpoint(initial_dir / "last.pth", checkpoint)
        write_summary(initial_dir, checkpoint, args.epochs)
        if run_dir.exists():
            raise FileExistsError(f"Another run already created {run_dir}; initialized files kept at {initial_dir}")
        initial_dir.rename(run_dir)
    write_summary(run_dir, checkpoint, args.epochs)
    print(f"Run: {run_dir}\nDataset: {config['dataset']}\nInitial weights SHA256: {init_hash}\nSplit SHA256: {split['split_sha256']}", flush=True)
    target_epoch = min(args.epochs, args.stop_after) if args.stop_after is not None else args.epochs
    for epoch in range(checkpoint["epoch"] + 1, target_epoch + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        if teacher is not None:
            teacher.eval()
        sums = {"total": 0.0, "ce": 0.0, "kd": 0.0}
        correct, count = 0, 0
        batches = tqdm(train_loader, desc=f"{args.role} epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(images)
            if teacher is None:
                ce_loss = nn.functional.cross_entropy(student_logits, labels)
                kd_loss = ce_loss.new_zeros(())
                total_loss = ce_loss
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                total_loss, ce_loss, kd_loss = distillation_loss(
                    student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA)
            if not torch.isfinite(total_loss).item():
                raise FloatingPointError("Non-finite training loss; previous epoch checkpoint remains available")
            total_loss.backward()
            optimizer.step()
            batch_count = labels.numel()
            for key, value in (("total", total_loss), ("ce", ce_loss), ("kd", kd_loss)):
                sums[key] += value.detach().item() * batch_count
            correct += int((student_logits.detach().argmax(1) == labels).sum().item())
            count += batch_count
            batches.set_postfix(total_loss=f"{sums['total']/count:.4f}", train_acc=f"{correct/count:.4f}")
        if count != len(training_data):
            raise RuntimeError("Training epoch did not cover every training sample")
        validation = evaluate_loader(model, val_loader, device)
        if validation["count"] != len(validation_data):
            raise RuntimeError("Validation did not cover every held-out sample")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"epoch": epoch, "train_total_loss": sums["total"] / count,
               "train_ce_loss": sums["ce"] / count, "train_kd_loss": sums["kd"] / count,
               "train_acc": correct / count, "val_loss": validation["loss"],
               "val_acc": validation["accuracy"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started}
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("Non-finite metrics; previous epoch checkpoint remains available")
        improved = row["val_acc"] > checkpoint["best_val_acc"]
        state_hash = model_state_sha256(model.state_dict())
        checkpoint = {
            "schema_version": 1, "config": config, "epoch": epoch,
            "model_state": model.state_dict(), "model_state_sha256": state_hash,
            "init_sha256": init_hash, "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng(train_generator, val_generator),
            "best_epoch": epoch if improved else checkpoint["best_epoch"],
            "best_val_acc": row["val_acc"] if improved else checkpoint["best_val_acc"],
            "best_val_loss": row["val_loss"] if improved else checkpoint["best_val_loss"],
            "best_model_state_sha256": state_hash if improved else checkpoint["best_model_state_sha256"],
            "last_metrics": row,
        }
        # last.pth is the epoch commit. If interrupted afterwards, resume repairs
        # the derived best.pth / metrics.csv / summary.json from that checkpoint.
        save_checkpoint(run_dir / "last.pth", checkpoint)
        if improved:
            save_checkpoint(run_dir / "best.pth", best_payload(checkpoint))
        rows.append(row)
        write_csv(run_dir / "metrics.csv", rows)
        write_summary(run_dir, checkpoint, args.epochs)
        print(f"epoch={epoch} val_acc={row['val_acc']:.6f} best_epoch={checkpoint['best_epoch']}", flush=True)
    state = "Complete" if checkpoint["epoch"] == args.epochs else "Stopped at committed epoch boundary"
    print(f"{state}: {run_dir / 'summary.json'}", flush=True)
    return checkpoint


if __name__ == "__main__":
    main()
