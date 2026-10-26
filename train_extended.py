"""Train a predeclared extension case without modifying the completed study.

The model, objective, data/RNG and atomic I/O helpers are reused unchanged. Main
seed0/1 students inherit their first ten committed epochs into a new directory;
the original checkpoints, logs and source are always read only.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import extended_protocol as protocol
import train as legacy
from models import build_model, distillation_loss
from kd_utils import (CSV_FIELDS, MEAN, STD, best_payload, capture_rng, evaluate_loader,
                      file_sha256, load_checkpoint, load_json, make_datasets,
                      model_state_sha256, restore_rng, save_checkpoint, save_json,
                      seed_everything, seed_worker, software_information,
                      validate_csv, verify_best, write_csv)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_log(path):
    """Read committed rows without repairing or otherwise writing a source run."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_FIELDS, "Unexpected training log columns")
        rows = []
        for index, raw in enumerate(reader, 1):
            require(None not in raw and all(v is not None for v in raw.values()), "Malformed log row")
            row = {key: int(value) if key == "epoch" else float(value) for key, value in raw.items()}
            require(row["epoch"] == index and all(np.isfinite(v) for v in row.values()),
                    "Nonsequential or nonfinite training log")
            require(0 <= row["train_acc"] <= 1 and 0 <= row["val_acc"] <= 1
                    and row["train_ce_loss"] >= 0 and row["val_loss"] >= 0
                    and row["train_kd_loss"] >= -1e-6 and row["seconds"] > 0
                    and row["lr"] == 0.001, "Invalid training log metrics")
            rows.append(row)
    return rows


def read_boundary(folder, expected_source, require_complete=True):
    """Strict, read-only validation of a new or legacy committed boundary."""
    folder = Path(folder).resolve(strict=True)
    config = load_json(folder / "config.json")
    summary = load_json(folder / "summary.json")
    last_path = folder / "last.pth"
    last_hash = file_sha256(last_path)
    last = load_checkpoint(last_path)
    require(last.get("config") == config, "last.pth and config.json differ")
    require(config.get("source") == expected_source, "Training source provenance differs")
    epoch = last.get("epoch")
    require(type(epoch) is int and 0 <= epoch <= config["epochs"], "Invalid committed epoch")
    if require_complete:
        require(epoch == config["epochs"], "Training has not completed its requested budget")
    require(model_state_sha256(last["model_state"]) == last.get("model_state_sha256"),
            "Last model tensor hash failed")
    require(last.get("init_sha256") == config.get("init_sha256"), "Initialization metadata differs")
    rows = read_log(folder / "metrics.csv")
    require(len(rows) == epoch and (not rows or rows[-1] == last["last_metrics"]),
            "Training log and last checkpoint differ")
    require(summary.get("status") == ("completed" if epoch == config["epochs"] else "in_progress")
            and summary.get("completed_epochs") == epoch
            and summary.get("requested_epochs") == config["epochs"], "Training summary budget differs")
    require(summary.get("last_checkpoint_sha256") == last_hash, "Last checkpoint file hash differs")
    for key, expected in (("init_sha256", config["init_sha256"]),
                          ("split_sha256", config["split_sha256"]),
                          ("best_epoch", last["best_epoch"] or None)):
        require(summary.get(key) == expected, f"Training summary disagrees: {key}")
    best, best_hash = None, None
    if epoch:
        chosen = max(rows, key=lambda row: row["val_acc"])
        require(chosen["epoch"] == last["best_epoch"] and chosen["val_acc"] == last["best_val_acc"],
                "Validation selection differs from logged maximum")
        best_path = folder / "best.pth"
        best_hash = file_sha256(best_path)
        best = load_checkpoint(best_path)
        state_hash = model_state_sha256(best["model_state"])
        require(best.get("config") == config and best.get("epoch") == chosen["epoch"]
                and best.get("best_epoch") == chosen["epoch"]
                and best.get("last_metrics") == chosen
                and best.get("best_val_acc") == chosen["val_acc"]
                and best.get("init_sha256") == config["init_sha256"], "Best checkpoint selection differs")
        require(best.get("model_state_sha256") == state_hash
                == best.get("best_model_state_sha256")
                == last.get("best_model_state_sha256")
                == summary.get("best_model_state_sha256"), "Best model tensor hash differs")
        require(summary.get("best_checkpoint_sha256") == best_hash
                and summary.get("best_val_acc") == chosen["val_acc"], "Best checkpoint file/summary differs")
    require(file_sha256(last_path) == last_hash, "Source checkpoint changed while being read")
    return best, last, summary, last_hash, best_hash, rows


def validate_case_config(config, allow_synthetic=False):
    require(config.get("study") == protocol.STUDY, "Not an extended-study checkpoint")
    spec = protocol.case_spec(config.get("case_id"))
    synthetic = config.get("dataset") == "synthetic"
    require(config.get("dataset") in ("CIFAR10", "synthetic"), "Unexpected data identity")
    require(allow_synthetic or not synthetic, "Synthetic verification is not a formal experiment")
    for key, value in (("role", spec["role"]), ("seed", spec["seed"]), ("group", spec["group"]),
                       ("temperature", spec["temperature"]), ("alpha", spec["alpha"]),
                       ("optimizer", legacy.OPTIMIZER), ("normalize_mean", MEAN),
                       ("normalize_std", STD), ("precision", "float32"), ("augmentation", "none")):
        require(config.get(key) == value, f"Case configuration differs: {key}")
    if not synthetic:
        require(not spec["reuse_legacy_teacher"], "Legacy teachers must be referenced, not retrained")
        require(config.get("epochs") == spec["epochs"] and config.get("batch_size") == 128
                and config.get("workers") == 2 and config.get("device") in ("cuda", "cuda:0")
                and config.get("deterministic") is True and config.get("tf32") is False
                and config.get("software", {}).get("cpu_threads") == 4
                and config.get("split", {}).get("split_seed") == 20261026
                and config.get("split", {}).get("train_size") == 45000
                and config.get("split", {}).get("val_size") == 5000,
                "Not the frozen formal extension configuration")
    return spec, synthetic


def validate_run(checkpoint_path, allow_synthetic=False):
    """Return (best,last,summary,best-file-sha) for one complete extension case."""
    path = Path(checkpoint_path).resolve(strict=True)
    require(path.name == "best.pth", "Use validation-selected best.pth")
    best, last, summary, _, digest, _ = read_boundary(path.parent, protocol.training_source())
    spec, synthetic = validate_case_config(best["config"], allow_synthetic)
    require(path.parent.name == spec["run_id"], "Case directory identity differs")
    config = best["config"]
    require(config.get("source") == protocol.training_source(), "Extension source changed")
    warm = config.get("warm_start")
    if warm is not None:
        source_path = Path(warm["checkpoint"]).resolve(strict=True)
        rebuilt = prepare_warm(source_path, config, config["init_sha256"], synthetic)[3]
        require(rebuilt == warm, "Warm-start dependency changed")
        if not synthetic:
            required = Path(config["project_root"]) / "runs" / spec["warm_start_run_id"] / "last.pth"
            require(source_path == required.resolve(), "Warm source is not the declared original run")
    if not synthetic:
        require((warm is not None) == (spec["warm_start_run_id"] is not None),
                "Main seed0/1 must retain their actual original ten-epoch prefix")
    if config["role"] == "kd":
        dependency = config.get("teacher")
        require(isinstance(dependency, dict), "KD teacher dependency is missing")
        teacher_path = Path(dependency["checkpoint"]).resolve(strict=True)
        teacher_best, _, teacher_hash = validate_teacher(teacher_path, config)
        expected_dependency = {"checkpoint": str(teacher_path), "weights_sha256": teacher_hash,
                               "model_state_sha256": teacher_best["model_state_sha256"], "seed": config["seed"]}
        require(dependency == expected_dependency, "KD teacher dependency metadata changed")
        if not synthetic:
            declared = protocol.teacher_checkpoint(Path(config["project_root"]), path.parent.parent, config["seed"])
            require(teacher_path == declared.resolve(), "KD uses an undeclared teacher path")
    else:
        require(config.get("teacher") is None, "Teacher/baseline must not have a teacher dependency")
    return best, last, summary, digest


def validate_teacher(path, expected):
    """Read-only dependency audit allowing student-specific T/alpha/budget."""
    path = Path(path).resolve(strict=True)
    require(path.name == "best.pth", "Teacher must be validation-selected best.pth")
    config = load_json(path.parent / "config.json")
    synthetic = expected["dataset"] == "synthetic"
    if config.get("study") == protocol.STUDY:
        best, _, summary, digest = validate_run(path, allow_synthetic=synthetic)
    else:
        best, _, summary, _, digest, _ = read_boundary(path.parent, legacy.training_source())
    require(config.get("role") == "teacher" and config.get("teacher") is None,
            "Dependency is not an independently trained teacher")
    if not synthetic:
        require(config.get("epochs") == 10 and summary.get("completed_epochs") == 10,
                "Formal teachers must complete ten epochs")
        require(config.get("temperature") == 2.0 and config.get("alpha") == 0.25,
                "Teacher recipe differs from the frozen default")
    for key in ("seed", "batch_size", "workers", "device", "precision", "dataset", "data_root",
                "split_sha256", "optimizer", "normalize_mean", "normalize_std", "software"):
        require(config.get(key) == expected[key], f"Teacher and student differ: {key}")
    return best, summary, digest


def load_teacher(path, expected, device):
    """Load an audited dependency and freeze all parameters and dropout behavior."""
    path = Path(path).resolve(strict=True)
    best, _, digest = validate_teacher(path, expected)
    config = best["config"]
    model = build_model("teacher").float().to(device)
    model.load_state_dict(best["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model, {"checkpoint": str(path), "weights_sha256": digest,
                   "model_state_sha256": best["model_state_sha256"], "seed": config["seed"]}


def prepare_warm(path, expected, init_hash, synthetic):
    path = Path(path).resolve(strict=True)
    require(path.name == "last.pth", "Continue the committed last.pth, not validation-selected best")
    best, last, _, digest, best_hash, rows = read_boundary(
        path.parent, legacy.training_source(), require_complete=not synthetic)
    config = last["config"]
    require(0 < last["epoch"] < expected["epochs"], "Warm boundary must precede the extended target epoch")
    require(best is not None, "Warm source has no selected model")
    if not synthetic:
        require(config.get("epochs") == last["epoch"] == 10, "Formal continuation starts after original epoch ten")
    for key in ("role", "seed", "batch_size", "workers", "device", "precision", "dataset", "data_root",
                "split_sha256", "optimizer", "temperature", "alpha", "normalize_mean", "normalize_std",
                "software", "teacher"):
        require(config.get(key) == expected[key], f"Warm source changes training math/runtime: {key}")
    require(config.get("init_sha256") == init_hash == last.get("init_sha256"), "Warm source has different initial weights")
    record = {"checkpoint": str(path), "weights_sha256": digest, "completed_epochs": last["epoch"],
              "original_config": config, "original_source": config["source"],
              "best_checkpoint_sha256": best_hash}
    return best, last, rows, record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, help="Synthetic verification only")
    parser.add_argument("--stop-after", type=int, help="Stop after a committed epoch; target budget is unchanged")
    args = parser.parse_args()
    spec = protocol.case_spec(args.case_id)
    if args.epochs is not None and not args.synthetic:
        parser.error("Formal case epoch budgets are fixed; --epochs is synthetic-only")
    args.epochs = args.epochs if args.epochs is not None else spec["epochs"]
    if args.epochs < 1 or args.workers < 0 or (args.stop_after is not None and not 0 <= args.stop_after <= args.epochs):
        parser.error("Invalid epoch/workers/stop-after budget")
    if spec["reuse_legacy_teacher"] and not args.synthetic:
        parser.error("Formal teacher seed0/1 already exist and must be reused")
    if spec["role"] != "kd" and args.teacher_checkpoint is not None:
        parser.error("Only KD has a teacher dependency")
    if args.warm_start_checkpoint is not None and spec["warm_start_run_id"] is None:
        parser.error("Only main seed0/1 students permit inherited legacy training")
    return args, spec


def main():
    args, spec = parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    require(device.type in ("cpu", "cuda"), "Only CPU and CUDA are supported")
    require(device.type != "cuda" or torch.cuda.is_available(), "Requested CUDA is unavailable")
    project = args.project_root.resolve(strict=True)
    output = (args.output_root or project / "runs_extended").resolve()
    data_root = (args.data_root or project / "data").resolve()
    for protected in (project / "runs", project / "results"):
        require(not output.is_relative_to(protected.resolve()), "Extended output must not modify original runs/results")
    if not args.synthetic:
        protocol.verify_manifest(project)
        require(args.workers == 2 and str(device) in ("cuda", "cuda:0"), "Formal extension uses CUDA/workers2")
    run_dir = output / spec["run_id"]
    require(not args.resume or run_dir.is_dir(), f"No extended case to resume: {run_dir}")
    require(args.resume or not run_dir.exists(), f"Refusing to overwrite {run_dir}; use --resume")
    training, validation, split = make_datasets(data_root, output / "splits", args.synthetic)
    expected = {"schema_version": 1, "study": protocol.STUDY, "case_id": spec["run_id"],
                "group": spec["group"], "role": spec["role"], "seed": spec["seed"],
                "dataset": "synthetic" if args.synthetic else "CIFAR10",
                "source_partition": "synthetic" if args.synthetic else "train", "epochs": args.epochs,
                "batch_size": 128, "workers": args.workers, "device": str(device), "precision": "float32",
                "data_root": str(data_root), "project_root": str(project),
                "split_seed": split["split_seed"], "split_sha256": split["split_sha256"],
                "split": {key: split[key] for key in ("split_seed", "train_size", "val_size", "split_sha256")},
                "normalize_mean": MEAN, "normalize_std": STD, "augmentation": "none",
                "optimizer": legacy.OPTIMIZER, "temperature": spec["temperature"], "alpha": spec["alpha"],
                "teacher": None, "warm_start": None, "deterministic": True, "tf32": False,
                "accuracy_unit": "fraction", "software": software_information(device),
                "source": protocol.training_source()}
    teacher = None
    if spec["role"] == "kd":
        teacher_path = args.teacher_checkpoint or protocol.teacher_checkpoint(project, output, spec["seed"], args.synthetic)
        if not args.synthetic:
            require(teacher_path.resolve() == protocol.teacher_checkpoint(project, output, spec["seed"]).resolve(),
                    "Formal KD must use the predeclared same-seed teacher")
        teacher, expected["teacher"] = load_teacher(teacher_path, expected, device)
    # Exactly the original initialization and paired shuffle/dropout RNG policy.
    seed_everything(spec["seed"])
    model = build_model(spec["role"])
    init_hash = model_state_sha256(model.state_dict())
    warm_path = args.warm_start_checkpoint
    if not args.synthetic and spec["warm_start_run_id"] is not None:
        required = project / "runs" / spec["warm_start_run_id"] / "last.pth"
        require(warm_path is None or warm_path.resolve() == required.resolve(), "Formal warm source differs from declared original run")
        warm_path = required
    warm = None
    if warm_path is not None:
        warm = prepare_warm(warm_path, expected, init_hash, args.synthetic)
        expected["warm_start"] = warm[3]
    config = {**expected, "init_sha256": init_hash}
    train_generator = torch.Generator().manual_seed(spec["seed"])
    val_generator = torch.Generator().manual_seed(spec["seed"] + 10000)
    options = dict(batch_size=128, num_workers=args.workers, pin_memory=device.type == "cuda",
                   worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(training, shuffle=True, generator=train_generator, **options)
    val_loader = DataLoader(validation, shuffle=False, generator=val_generator, **options)
    model = model.float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    if args.resume:
        require(load_json(run_dir / "config.json") == config, "Extended resume configuration/dependencies changed")
        checkpoint = load_checkpoint(run_dir / "last.pth")
        require(checkpoint.get("config") == config and checkpoint.get("init_sha256") == init_hash,
                "Extended last checkpoint configuration differs")
        require(0 <= checkpoint["epoch"] <= args.epochs, "Invalid saved epoch")
        require(model_state_sha256(checkpoint["model_state"]) == checkpoint["model_state_sha256"],
                "Extended last model hash failed")
        rows = validate_csv(run_dir / "metrics.csv", checkpoint)
        verify_best(run_dir / "best.pth", checkpoint)
    else:
        output.mkdir(parents=True, exist_ok=True)
        initial = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.initializing-", dir=output))
        if warm is None:
            rows = []
            checkpoint = {"schema_version": 1, "config": config, "epoch": 0,
                          "model_state": model.state_dict(), "model_state_sha256": init_hash,
                          "init_sha256": init_hash, "optimizer_state": optimizer.state_dict(),
                          "rng_state": capture_rng(train_generator, val_generator), "best_epoch": 0,
                          "best_val_acc": -1.0, "best_val_loss": None,
                          "best_model_state_sha256": None, "last_metrics": None}
        else:
            old_best, old_last, rows, _ = warm
            checkpoint = {**old_last, "config": config}
            save_checkpoint(initial / "best.pth", {**old_best, "config": config})
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
    print(f"Case: {spec['run_id']}; completed={checkpoint['epoch']}/{args.epochs}; T={spec['temperature']}; alpha={spec['alpha']}", flush=True)
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
            require(bool(torch.isfinite(loss)), "Non-finite training loss; previous epoch remains committed")
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
