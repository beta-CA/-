"""Predeclared three-hour extension; the completed ten-epoch study is immutable."""
import hashlib
import json
from pathlib import Path

STUDY = "distilling_extended_v1"
MAIN_SEEDS = (0, 1, 2)
ABLATION_SEEDS = (0, 1)
TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
ALPHAS = (0.1, 0.25, 0.5, 0.75)
DEFAULT_TEMPERATURE = 2.0
DEFAULT_ALPHA = 0.25
STUDENT_EPOCHS = 20
TEACHER_EPOCHS = 10
ROOT = Path(__file__).resolve().parent


def _case(run_id, role, seed, group="main", temperature=2.0, alpha=0.25,
          warm_start_run_id=None, reuse_legacy_teacher=False):
    return {"run_id": run_id, "role": role, "seed": seed,
            "epochs": TEACHER_EPOCHS if role == "teacher" else STUDENT_EPOCHS,
            "temperature": float(temperature), "alpha": float(alpha), "group": group,
            "warm_start_run_id": warm_start_run_id,
            "reuse_legacy_teacher": reuse_legacy_teacher}


def training_cases():
    cases = [_case("teacher_seed2", "teacher", 2)]
    for seed in MAIN_SEEDS:
        for role in ("baseline", "kd"):
            run_id = f"{role}_seed{seed}"
            cases.append(_case(run_id, role, seed,
                               warm_start_run_id=run_id if seed in (0, 1) else None))
    for temperature in TEMPERATURES:
        if temperature != DEFAULT_TEMPERATURE:
            for seed in ABLATION_SEEDS:
                cases.append(_case(f"kd_T{int(temperature)}_seed{seed}", "kd", seed,
                                   "temperature", temperature=temperature))
    for alpha in ALPHAS:
        if alpha != DEFAULT_ALPHA:
            for seed in ABLATION_SEEDS:
                token = str(alpha).replace(".", "p")
                cases.append(_case(f"kd_A{token}_seed{seed}", "kd", seed,
                                   "alpha", alpha=alpha))
    assert len(cases) == 19 and len({item["run_id"] for item in cases}) == 19
    return cases


def all_cases():
    return [_case(f"teacher_seed{seed}", "teacher", seed, reuse_legacy_teacher=True)
            for seed in (0, 1)] + training_cases()


def case_spec(case_id):
    matches = [item for item in all_cases() if item["run_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Case is not in the fixed extension: {case_id}")
    return dict(matches[0])


def teacher_checkpoint(project_root, output_root, seed, synthetic=False):
    project_root, output_root = Path(project_root), Path(output_root)
    if synthetic or seed == 2:
        return output_root / f"teacher_seed{seed}" / "best.pth"
    return project_root / "runs" / f"teacher_seed{seed}" / "best.pth"


def temperature_case_id(temperature, seed):
    return f"kd_seed{seed}" if float(temperature) == 2.0 else f"kd_T{int(temperature)}_seed{seed}"


def alpha_case_id(alpha, seed):
    return f"kd_seed{seed}" if float(alpha) == 0.25 else f"kd_A{str(float(alpha)).replace('.', 'p')}_seed{seed}"


def training_source():
    from train import training_source as legacy_source
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
             for name in ("train_extended.py", "extended_protocol.py")}
    return {"legacy": legacy_source(), "extension_files_sha256": files}


def study_document():
    return {
        "study": STUDY, "main_seeds": list(MAIN_SEEDS), "ablation_seeds": list(ABLATION_SEEDS),
        "temperature_values": list(TEMPERATURES), "alpha_values": list(ALPHAS),
        "teacher_epochs": 10, "student_epochs": 20,
        "data": {"dataset": "CIFAR10", "split_seed": 20261026,
                 "train_size": 45000, "validation_size": 5000, "test_size": 10000},
        "training": {"batch_size": 128, "workers": 2, "device": "cuda",
                     "precision": "float32", "cpu_threads": 4, "optimizer": "Adam",
                     "learning_rate": 0.001, "augmentation": "none",
                     "normalize_mean": [0.485, 0.456, 0.406],
                     "normalize_std": [0.229, 0.224, 0.225]},
        "selection": "highest validation accuracy; earliest epoch wins ties",
        "case_count": 21, "new_or_continued_training_cases": 19,
        "new_student_epochs": 320, "new_teacher_epochs": 10,
        "previously_completed_epochs": 60, "total_actual_trained_epochs_after_extension": 390,
        "rules": [
            "Reuse original teacher seed0/1, independently train teacher seed2 for ten epochs.",
            "Continue main student seed0/1 from epoch10, preserving optimizer/RNG/log history and original best.",
            "Train all other students fresh for twenty epochs with paired student initialization per seed.",
            "T2/alpha0.25 is the shared main default and is not retrained for either ablation table.",
            "Vary one factor at a time; no T-by-alpha joint search.",
            "All extension runs finish before final testing; no test-based recipe selection.",
            "Original test results have already been inspected: this is an exploratory extension.",
            "Original source/checkpoints/logs/evaluations remain unchanged."
        ], "cases": all_cases()}


def verify_manifest(project_root):
    path = Path(project_root) / "configs" / "extended_experiments.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != study_document():
        raise ValueError("Extended experiment manifest differs from the fixed protocol")
    return actual


def estimated_budget(project_root):
    summary = json.loads((Path(project_root) / "results" / "summary.json").read_text(encoding="utf-8"))
    groups = {row["role"]: row for row in summary["groups"]}
    per_epoch = {role: row["training_seconds_total_mean"] / 10 for role, row in groups.items()}
    new_seconds = 10 * per_epoch["teacher"] + 40 * per_epoch["baseline"] + 280 * per_epoch["kd"]
    old_seconds = sum(2 * row["training_seconds_total_mean"] for row in groups.values())
    return {"per_epoch_seconds_from_completed_study": per_epoch,
            "estimated_additional_training_validation_minutes": new_seconds / 60,
            "completed_training_validation_minutes": old_seconds / 60,
            "estimated_combined_training_validation_minutes": (old_seconds + new_seconds) / 60,
            "note": "Measured-rate extrapolation; startup/checkpoint/test/plot and future performance variation are extra."}
