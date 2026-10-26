"""Reuse the PyTorch tutorial networks and its logit-distillation objective.

Method: Hinton, Vinyals and Dean, arXiv:1503.02531 (2015).
Implementation source: PyTorch tutorial by Alexandros Chariton (BSD-3-Clause),
not the original paper authors' implementation. See vendor/pytorch_tutorial/.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from vendor.pytorch_tutorial.models import DeepNN, LightNN

ROOT = Path(__file__).resolve().parent


def build_model(role):
    if role not in ("teacher", "baseline", "kd"):
        raise ValueError(f"Unknown role: {role}")
    return DeepNN(num_classes=10) if role == "teacher" else LightNN(num_classes=10)


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.25):
    """Return total, hard-label CE, and T² KL(teacher || student).

F.kl_div implements the tutorial's sum(p_teacher * (log p_teacher - log p_student))
with zero-probability handling. The teacher is detached, including for standalone
calls to this function. The same temperature is used on both distributions.
"""
    if not (0.0 < temperature < float("inf")) or not (0.0 <= alpha <= 1.0):
        raise ValueError("temperature must be finite and positive; alpha must be in [0,1]")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Teacher/student logits must have matching [batch, classes] shapes")
    ce = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature ** 2
    total = (1.0 - alpha) * ce + alpha * kd
    return total, ce, kd


def source_information():
    """Verify vendored bytes, then describe the implementation actually executed."""
    vendor = ROOT / "vendor" / "pytorch_tutorial"
    manifest_path = vendor / "UPSTREAM.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Vendored source differs from UPSTREAM.json: {name}")
    return {
        "method_paper": "https://arxiv.org/abs/1503.02531",
        "implementation_repository": manifest["repository"],
        "implementation_commit": manifest["commit"],
        "implementation_source_path": manifest["source_path"],
        "vendor_files_sha256": manifest["files_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "models_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loss": "(1-alpha)*CE + alpha*T^2*KL(teacher||student); stable F.kl_div",
    }
