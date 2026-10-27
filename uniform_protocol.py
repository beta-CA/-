"""Separate, fixed 20-epoch study; published historical experiments stay intact."""
import hashlib
import json
from pathlib import Path

STUDY = "distilling_uniform20_v2"
TRAINING_EPOCHS = 20
TEACHER_EPOCHS = STUDENT_EPOCHS = TRAINING_EPOCHS
MAIN_SEEDS = (0, 1, 2)
ABLATION_SEEDS = (0, 1)
TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
ALPHAS = (0.1, 0.25, 0.5, 0.75)
DEFAULT_TEMPERATURE = 2.0
DEFAULT_ALPHA = 0.25
ROOT = Path(__file__).resolve().parent
RUNS_DIRECTORY = "runs_uniform20"
RESULTS_DIRECTORY = "results_uniform20"
MANIFEST = "uniform20_experiments.json"
PROTECTED_DIRECTORIES = ("runs", "results", "runs_extended", "results_extended",
                         "data", "vendor", "configs", "paper_figures")


def _case(run_id, role, seed, group="main", temperature=2.0, alpha=0.25):
    return {"run_id": run_id, "role": role, "seed": seed,
            "epochs": TRAINING_EPOCHS, "temperature": float(temperature),
            "alpha": float(alpha), "group": group,
            "warm_start_run_id": None, "reuse_legacy_teacher": False}


def training_cases():
    """All 21 cases start fresh, with teachers first and paired students after."""
    cases = [_case(f"teacher_seed{s}", "teacher", s) for s in MAIN_SEEDS]
    for seed in MAIN_SEEDS:
        cases.extend(_case(f"{role}_seed{seed}", role, seed) for role in ("baseline", "kd"))
    for temperature in TEMPERATURES:
        if temperature != DEFAULT_TEMPERATURE:
            cases.extend(_case(f"kd_T{int(temperature)}_seed{s}", "kd", s,
                               "temperature", temperature=temperature) for s in ABLATION_SEEDS)
    for alpha in ALPHAS:
        if alpha != DEFAULT_ALPHA:
            token = str(alpha).replace(".", "p")
            cases.extend(_case(f"kd_A{token}_seed{s}", "kd", s, "alpha", alpha=alpha)
                         for s in ABLATION_SEEDS)
    assert len(cases) == 21 and len({x["run_id"] for x in cases}) == 21
    return cases


def all_cases():
    return training_cases()


def case_spec(case_id):
    matches = [x for x in all_cases() if x["run_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Case is not in the uniform-20 study: {case_id}")
    return dict(matches[0])


def teacher_checkpoint(project_root, output_root, seed, synthetic=False):
    del synthetic
    if seed not in MAIN_SEEDS:
        raise ValueError(f"Teacher seed is not declared: {seed}")
    output = validate_output_root(project_root, output_root)
    return output / f"teacher_seed{seed}" / "best.pth"


def validate_output_root(project_root, output_root):
    """Reject historical data/results and either direction of path overlap."""
    project = Path(project_root).resolve(strict=True)
    output = Path(output_root).resolve()
    if output == project or project.is_relative_to(output):
        raise ValueError("Use a separate output directory, not the project or its ancestor")
    for name in PROTECTED_DIRECTORIES:
        protected = (project / name).resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"Uniform-20 output overlaps protected directory: {name}")
    return output


def temperature_case_id(temperature, seed):
    return f"kd_seed{seed}" if float(temperature) == 2.0 else f"kd_T{int(temperature)}_seed{seed}"


def alpha_case_id(alpha, seed):
    return f"kd_seed{seed}" if float(alpha) == 0.25 else f"kd_A{str(float(alpha)).replace('.', 'p')}_seed{seed}"


def training_source():
    from train import training_source as legacy_source
    names = ("train_uniform20.py", "uniform_protocol.py", "uniform20_recovery.py",
             "train_extended.py", "extended_recovery.py")
    return {"legacy": legacy_source(),
            "uniform20_files_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in names}}


def study_document():
    cases = all_cases()
    return {"study": STUDY, "training_epochs": TRAINING_EPOCHS,
            "teacher_epochs": TRAINING_EPOCHS, "student_epochs": TRAINING_EPOCHS,
            "main_seeds": list(MAIN_SEEDS), "ablation_seeds": list(ABLATION_SEEDS),
            "temperature_values": list(TEMPERATURES), "alpha_values": list(ALPHAS),
            "data": {"dataset": "CIFAR10", "split_seed": 20261026,
                     "train_size": 45000, "validation_size": 5000, "test_size": 10000},
            "training": {"batch_size": 128, "workers": 2, "device": "cuda",
                         "precision": "float32", "cpu_threads": 4, "optimizer": "Adam",
                         "learning_rate": 0.001, "augmentation": "none",
                         "normalize_mean": [0.485, 0.456, 0.406],
                         "normalize_std": [0.229, 0.224, 0.225]},
            "selection": "highest validation accuracy; earliest epoch wins ties",
            "case_count": len(cases), "fresh_training_cases": len(cases),
            "teacher_training_epochs": 3 * TRAINING_EPOCHS,
            "student_training_epochs": 18 * TRAINING_EPOCHS,
            "total_training_epochs": len(cases) * TRAINING_EPOCHS,
            "runs_directory": RUNS_DIRECTORY, "results_directory": RESULTS_DIRECTORY,
            "rules": [
                "Train every teacher, baseline and KD case afresh for twenty epochs in a separate study.",
                "Do not inherit historical checkpoints, optimizer states, logs or evaluations.",
                "A KD student uses only the completed same-seed uniform-20 teacher from its own runs root.",
                "Freeze the validation-selected teacher; best.pth may precede epoch twenty.",
                "T2/alpha0.25 is shared across main and ablation tables, not trained twice.",
                "Use paired student initialization and data order; vary one factor at a time.",
                "Finish every planned training case before final evaluation; no test-based recipe selection.",
                "Historical test results have already been inspected; this remains exploratory coursework.",
                "Preserve historical source files, checkpoints, logs, results and published v1 release."
            ], "cases": cases}


def verify_manifest(project_root):
    path = Path(project_root) / "configs" / MANIFEST
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != study_document():
        raise ValueError("Uniform-20 manifest differs from the fixed protocol")
    return actual


def estimated_budget(project_root):
    """Use historical measured speeds only, never historical model results."""
    path = Path(project_root) / "results" / "summary.json"
    if not path.is_file():
        return {"available": False, "total_training_epochs": 420,
                "estimated_training_validation_minutes": None,
                "note": "No historical speed record is present; run time is not yet estimated."}
    summary = json.loads(path.read_text(encoding="utf-8"))
    groups = {row["role"]: row for row in summary["groups"]}
    speeds = {role: float(groups[role]["training_seconds_total_mean"]) / 10
              for role in ("teacher", "baseline", "kd")}
    counts = {role: sum(x["epochs"] for x in all_cases() if x["role"] == role)
              for role in speeds}
    seconds = sum(speeds[role] * counts[role] for role in speeds)
    return {"available": True, "per_epoch_seconds_from_historical_study": speeds,
            "fresh_epochs_by_role": counts, "total_training_epochs": sum(counts.values()),
            "estimated_training_validation_minutes": seconds / 60,
            "note": "Measured-rate extrapolation for this new study only; startup, saving, testing, plots and future performance variation are extra."}
