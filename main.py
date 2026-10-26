"""Check the configured environment. Formal training uses Run All 6 Experiments."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from pathlib import Path
import platform
import sys


def main():
    import matplotlib
    import numpy as np
    import torch
    import torchvision
    from models import build_model, distillation_loss, source_information
    from prepare_data import prepare_data

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Python environment with a compatible CUDA-enabled PyTorch installation.")
    device = torch.device("cuda")
    torch.manual_seed(0)
    teacher, student = build_model("teacher").to(device), build_model("baseline").to(device)
    teacher.eval().requires_grad_(False)
    images, labels = torch.randn(2, 3, 32, 32, device=device), torch.tensor([0, 1], device=device)
    with torch.no_grad():
        targets = teacher(images)
    loss, ce, kd = distillation_loss(student(images), targets, labels)
    loss.backward()
    if not torch.isfinite(loss).item() or not any(p.grad is not None for p in student.parameters()):
        raise RuntimeError("CUDA distillation gradient check failed")
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Frozen teacher unexpectedly acquired gradients")
    root = Path(__file__).resolve().parent
    data_root = root / "data"
    # Preparation is a separate explicit action; this entry only verifies existing files.
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity
    entries = list(CIFAR10.train_list) + list(CIFAR10.test_list) + [
        (CIFAR10.meta["filename"], CIFAR10.meta["md5"])]
    data_verified = all(check_integrity(str(data_root / CIFAR10.base_folder / name), checksum)
                        for name, checksum in entries)
    report = {"python_executable": sys.executable, "python": platform.python_version(),
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
              "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
              "student_parameters": sum(p.numel() for p in student.parameters()),
              "cuda_forward_backward": "passed", "teacher_frozen": True,
              "project_local_cifar10_md5_verified": data_verified,
              "source": source_information(), "formal_training_started": False}
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nEnvironment check complete. No formal training was started.")
    print("To train: run run_all_experiments.py, then run_extended_experiments.py.")


if __name__ == "__main__":
    main()
