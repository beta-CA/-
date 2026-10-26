"""File-only regression checks for Windows initialization publication recovery.

Uses copies of the already-saved real initialization. Never trains or tests a
model. All test copies remain inside the dedicated verification directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(project))
    import extended_recovery as recovery
    from evaluate_extended import validate_training_matrix
    from kd_utils import file_sha256
    from run_extended_experiments import run_logged
    work = Path(__file__).resolve().parent / "_verification" / ("windows_recovery_" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True)
    checks = {}

    def hashes(folder):
        return {p.relative_to(folder).as_posix(): file_sha256(p)
                for p in folder.rglob("*") if p.is_file()}

    original = hashes(source)

    def candidate(label, token="test"):
        root = work / label
        root.mkdir(exist_ok=True)
        path = root / f".baseline_seed0.initializing-{token}"
        shutil.copytree(source, path)
        return root, path

    def refused(call):
        try:
            call()
        except (ValueError, OSError, RuntimeError):
            return
        raise AssertionError("Invalid recovery was accepted")

    root, initial = candidate("normal")
    result = recovery.recover_initialization("baseline_seed0", project, root)
    assert result["recovered"] and result["epoch"] == 10
    target = root / "baseline_seed0"
    assert all(file_sha256(target / name) == digest for name, digest in original.items())
    assert hashes(initial) == original
    assert recovery.recover_initialization("baseline_seed0", project, root)["reason"] == "already_recovered"
    # A completed recovery remains safe to inspect after evaluation files appear.
    for name in ("evaluation.json", "predictions.csv", "confusion_matrix.csv"):
        (target / name).write_text("verification placeholder only\n", encoding="utf-8")
    assert recovery.recover_initialization("baseline_seed0", project, root)["reason"] == "already_recovered"
    checks["byte_exact_recovery_idempotence_and_later_evaluation_files"] = True

    root2, initial2 = candidate("interrupted")
    real_copy = recovery.shutil.copyfileobj
    calls = 0

    def interrupted_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Injected copy interruption for verification")
        return real_copy(*args, **kwargs)

    recovery.shutil.copyfileobj = interrupted_copy
    try:
        refused(lambda: recovery.recover_initialization("baseline_seed0", project, root2))
    finally:
        recovery.shutil.copyfileobj = real_copy
    marker = root2 / "baseline_seed0" / recovery.MARKER_NAME
    assert json.loads(marker.read_text(encoding="utf-8"))["complete"] is False
    assert recovery.recover_initialization("baseline_seed0", project, root2)["recovered"]
    assert all(file_sha256(root2 / "baseline_seed0" / name) == digest for name, digest in original.items())
    assert hashes(initial2) == original
    checks["interrupted_copy_resumes_from_own_incomplete_marker"] = True

    root3, _ = candidate("existing_target")
    existing = root3 / "baseline_seed0"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    before = hashes(existing)
    assert not recovery.recover_initialization("baseline_seed0", project, root3)["recovered"]
    assert hashes(existing) == before
    checks["existing_target_without_marker_is_untouched"] = True

    root4, _ = candidate("multiple", "one")
    shutil.copytree(source, root4 / ".baseline_seed0.initializing-two")
    refused(lambda: recovery.recover_initialization("baseline_seed0", project, root4))
    assert not (root4 / "baseline_seed0").exists()
    checks["ambiguous_sources_rejected_before_target_creation"] = True

    root5, bad = candidate("corrupt")
    (bad / "last.pth").write_bytes(b"corrupted verification fixture")
    try:
        recovery.recover_initialization("baseline_seed0", project, root5)
    except Exception:
        pass
    else:
        raise AssertionError("Corrupt initialization accepted")
    assert not (root5 / "baseline_seed0").exists()
    checks["corrupt_source_rejected_before_target_creation"] = True

    root6, _ = candidate("conflicting_target")
    calls = 0
    recovery.shutil.copyfileobj = interrupted_copy
    try:
        refused(lambda: recovery.recover_initialization("baseline_seed0", project, root6))
    finally:
        recovery.shutil.copyfileobj = real_copy
    conflict = root6 / "baseline_seed0"
    (conflict / "config.json").write_text("{}\n", encoding="utf-8")
    conflict_before = hashes(conflict)
    refused(lambda: recovery.recover_initialization("baseline_seed0", project, root6))
    assert hashes(conflict) == conflict_before
    checks["conflicting_partial_target_rejected_without_writes"] = True

    try:
        validate_training_matrix(project, root)
    except ValueError as error:
        assert "unexpected=[]" in str(error), str(error)
    else:
        raise AssertionError("Incomplete matrix accepted")
    extra = root / "unexpected"
    extra.mkdir()
    (extra / "config.json").write_text("{}", encoding="utf-8")
    try:
        validate_training_matrix(project, root)
    except ValueError as error:
        assert "unexpected=['unexpected']" in str(error), str(error)
    else:
        raise AssertionError("Unexpected visible case accepted")
    checks["matrix_ignores_initialization_backup_but_rejects_unknown_case"] = True

    log = work / "captured_error.log"
    code = run_logged([sys.executable, "-c", "import sys; print('captured verification error'); sys.exit(3)"], project, log)
    assert code == 3 and "captured verification error" in log.read_text(encoding="utf-8")
    checks["runner_preserves_failure_exit_code_and_console_log"] = True
    assert hashes(source) == original
    checks["actual_saved_initialization_unchanged"] = True
    report = {"complete": True, "all_checks_passed": all(checks.values()), "checks": checks,
              "training_epochs_executed": 0, "official_test_used": False,
              "artifact_directory": str(work), "real_source_unchanged": True}
    report_path = Path(__file__).resolve().parent / "reports" / "windows_recovery_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
