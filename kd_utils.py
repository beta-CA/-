"""Shared deterministic I/O and data helpers reused from the Cutout project.
Only the official CIFAR-10 training partition is used for train/validation.
Synthetic fixtures are explicitly separate from any formal experiment.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets, transforms

SPLIT_SEED = 20261026
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CSV_FIELDS = ["epoch", "train_total_loss", "train_ce_loss", "train_kd_loss", "train_acc",
              "val_loss", "val_acc", "lr", "seconds"]

@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Return sample-weighted loss, accuracy fraction and count; restore mode."""
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.numel()
    finally:
        model.train(was_training)
    if not count:
        raise ValueError("Cannot evaluate an empty loader")
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def model_state_sha256(state):
    """Hash tensor names, dtypes, shapes and bytes, independent of .pth format."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(metadata.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    """Load this project's trusted checkpoint, including Python/NumPy RNG state."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def atomic_write(path, writer, binary=False):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb" if binary else "w", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False,
                                         **({} if binary else {"encoding": "utf-8", "newline": ""})) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, lambda stream: json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False))


def save_checkpoint(path, value):
    atomic_write(path, lambda stream: torch.save(value, stream), binary=True)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def capture_rng(train_generator, val_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_generator": train_generator.get_state(), "val_generator": val_generator.get_state()}


def restore_rng(state, train_generator, val_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device inventory differs from the saved RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator.set_state(state["train_generator"])
    val_generator.set_state(state["val_generator"])


def write_csv(path, rows):
    def writer(stream):
        output = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        output.writeheader()
        output.writerows(rows)
    atomic_write(path, writer)


def validate_csv(path, checkpoint):
    """last.pth commits an epoch; recover its row if interrupted before CSV save."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("Unexpected metrics.csv columns")
        rows = []
        for expected_epoch, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed metrics.csv row")
            parsed = {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
            if parsed["epoch"] != expected_epoch or not all(np.isfinite(v) for v in parsed.values()):
                raise ValueError("metrics.csv has invalid or non-sequential rows")
            rows.append(parsed)
    epoch = checkpoint["epoch"]
    if len(rows) == epoch - 1 and epoch > 0:
        rows.append(checkpoint["last_metrics"])
        write_csv(path, rows)
        print(f"Recovered committed epoch {epoch} CSV row from last.pth", flush=True)
    if len(rows) != epoch or (epoch and rows[-1] != checkpoint["last_metrics"]):
        raise ValueError("metrics.csv and last.pth do not describe the same epoch boundary")
    if epoch:
        selected = max(rows, key=lambda row: row["val_acc"])
        if selected["epoch"] != checkpoint["best_epoch"] or selected["val_acc"] != checkpoint["best_val_acc"]:
            raise ValueError("CSV validation maximum disagrees with checkpoint selection")
    return rows


def best_payload(checkpoint):
    best = {key: checkpoint[key] for key in (
        "schema_version", "model_state", "config", "epoch", "best_epoch", "best_val_acc",
        "best_val_loss", "model_state_sha256", "best_model_state_sha256", "init_sha256", "last_metrics")}
    best["val_acc"] = checkpoint["last_metrics"]["val_acc"]
    best["val_loss"] = checkpoint["last_metrics"]["val_loss"]
    return best


def verify_best(path, checkpoint):
    if checkpoint["best_epoch"] == 0:
        if Path(path).exists():
            raise ValueError("Unexpected best.pth before the first completed epoch")
        return
    valid = False
    if Path(path).exists():
        best = load_checkpoint(path)
        valid = (best["config"] == checkpoint["config"] and
                 best["epoch"] == checkpoint["best_epoch"] and
                 best["best_val_acc"] == checkpoint["best_val_acc"] and
                 model_state_sha256(best["model_state"]) == checkpoint["best_model_state_sha256"])
    if not valid:
        # A crash may happen after last.pth is committed but before best.pth.
        if checkpoint["epoch"] != checkpoint["best_epoch"]:
            raise ValueError("best.pth is missing or inconsistent; cannot reconstruct an older best model")
        if model_state_sha256(checkpoint["model_state"]) != checkpoint["best_model_state_sha256"]:
            raise ValueError("Cannot recover best.pth: current model hash differs")
        save_checkpoint(path, best_payload(checkpoint))
        print("Recovered best.pth from the committed last.pth", flush=True)


def build_transform():
    """Use the tutorial's fixed preprocessing, with no stochastic augmentation."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def software_information(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cpu_threads": torch.get_num_threads(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}


def ensure_split(split_root, targets):
    """Fully verify the persistent stratified split, including every sample index."""
    split_root = Path(split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(targets, dtype="<i8")
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected official CIFAR-10 train partition: 50,000 samples, 5,000 per class")
    generator = np.random.default_rng(SPLIT_SEED)
    val_indices = sorted(np.concatenate([
        generator.permutation(np.flatnonzero(labels == cls))[:500] for cls in range(10)
    ]).tolist())
    validation = set(val_indices)
    train_indices = [index for index in range(50000) if index not in validation]
    content = {"schema_version": 1, "dataset": "CIFAR10", "source_partition": "train",
               "split_seed": SPLIT_SEED, "val_per_class": 500,
               "train_size": 45000, "val_size": 5000,
               "targets_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
               "train_indices": train_indices, "val_indices": val_indices}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content["split_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = split_root / f"split_cifar10_seed{SPLIT_SEED}.json"
    if path.exists():
        if load_json(path) != content:
            raise ValueError(f"Split index/metadata/hash verification failed: {path}")
    else:
        save_json(path, content)
    return content


def make_datasets(data_root, split_root, synthetic=False):
    if synthetic:
        # Independent generators leave student initialization and dropout RNG unchanged.
        generator = torch.Generator().manual_seed(2026102601)
        images = torch.randn(320, 3, 32, 32, generator=generator)
        labels = torch.arange(320) % 10
        split = {"schema_version": 1, "dataset": "synthetic", "source_partition": "synthetic",
                 "split_seed": 2026102601, "train_size": 256, "val_size": 64,
                 "fixture": "normal_float32_tensor_seed2026102601_arange_mod10"}
        canonical = json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
        split["split_sha256"] = hashlib.sha256(canonical + images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        return TensorDataset(images[:256], labels[:256]), TensorDataset(images[256:], labels[256:]), split
    # This code never instantiates train=False or reads official test performance.
    official = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=build_transform())
    split = ensure_split(split_root, official.targets)
    return Subset(official, split["train_indices"]), Subset(official, split["val_indices"]), split
