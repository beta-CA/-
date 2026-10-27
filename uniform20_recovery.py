"""Publish a fully checked initialization without renaming its directory.

This Windows recovery helper never trains, edits an original run, or deletes
the failed initialization. Checkpoint/source formats remain unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import uuid


MARKER_NAME = ".uniform20_initialization_recovery.json"
HELPER_ID = "uniform20_initialization_recovery_v2"
BASE_FILES = ("config.json", "metrics.csv", "last.pth", "summary.json")
EVALUATION_FILES = ("evaluation.json", "predictions.csv", "confusion_matrix.csv")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_child(path, root):
    """Only direct, ordinary children of the resolved requested directory."""
    path = Path(path)
    _require(not path.is_symlink() and not path.is_junction(),
             f"Recovery refuses links/junctions: {path}")
    resolved = path.resolve()
    _require(resolved.parent == root and resolved.name == path.name,
             f"Recovery path escapes its requested directory: {path}")
    return resolved


def _owned_temporary(name):
    return bool(re.fullmatch(r"\.extended-recovery-[0-9a-f]{32}\.tmp", name))


def _write_marker(target, value):
    temporary = _safe_child(target / f".extended-recovery-{uuid.uuid4().hex}.tmp", target)
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target / MARKER_NAME)


def _read_candidate(folder, project, case_id, allow_synthetic):
    import uniform_protocol as protocol
    from kd_utils import file_sha256
    from train_uniform20 import read_boundary, validate_case_config

    for name in BASE_FILES:
        _require(_safe_child(folder / name, folder).is_file(),
                 f"Incomplete initialization: {folder / name}")
    best, last, _, _, _, _ = read_boundary(
        folder, protocol.training_source(), require_complete=False)
    config = last["config"]
    spec, _ = validate_case_config(config, allow_synthetic=allow_synthetic)
    _require(spec["run_id"] == config["case_id"] == case_id,
             "Initialization belongs to another case")
    _require(Path(config["project_root"]).resolve(strict=True) == project,
             "Initialization belongs to another project")
    epoch = last["epoch"]
    warm = config.get("warm_start")
    if warm is None:
        _require(epoch == 0 and best is None,
                 "Fresh recovery accepts only the untrained initialization")
        names = BASE_FILES
    else:
        _require(isinstance(warm, dict) and type(warm.get("completed_epochs")) is int
                 and epoch == warm["completed_epochs"] and epoch > 0,
                 "Warm recovery accepts only its inherited initialization boundary")
        _require(spec["warm_start_run_id"] == case_id,
                 "This case does not permit an inherited initialization")
        names = (*BASE_FILES, "best.pth")
    entries = list(folder.iterdir())
    _require({p.name for p in entries} == set(names),
             f"Initialization has missing or unexpected files: {folder}")
    for entry in entries:
        _require(_safe_child(entry, folder).is_file(),
                 f"Initialization contains a non-file: {entry}")
    hashes = {name: file_sha256(folder / name) for name in names}
    return epoch, hashes


def recover_initialization(case_id, project_root, runs_root, allow_synthetic=False):
    """Copy one validated, unpublished initialization to its final directory.

    A normal existing case is left alone. An interrupted helper copy is resumed
    only with its matching marker and matching already-published file hashes.
    Completed recovered cases may have subsequently advanced their training;
    they are validated and never rolled back. Private partial copy files and
    the complete source directory are retained rather than deleted.
    """
    import uniform_protocol as protocol
    from kd_utils import file_sha256, load_json
    from train_uniform20 import read_boundary, validate_case_config

    spec = protocol.case_spec(case_id)
    _require(spec["run_id"] == case_id, "Invalid recovery case identity")
    project = Path(project_root).resolve(strict=True)
    requested_root = Path(runs_root)
    _require(not requested_root.is_symlink() and not requested_root.is_junction(),
             "Recovery run root cannot be a link/junction")
    root = requested_root.resolve()
    _require(root != project, "Recovery requires a separate run directory")
    for name in ("runs", "results", "runs_extended", "results_extended", "data"):
        protected = (project / name).resolve()
        _require(not root.is_relative_to(protected),
                 f"Recovery cannot write into original {name}")
    target = _safe_child(root / case_id, root)
    result = {"source": None, "target": str(target), "epoch": None,
              "hashes": {}, "recovered": False}
    if not root.exists():
        return {**result, "reason": "no_initialization"}
    _require(root.is_dir(), "Recovery run root is not a directory")
    marker_path = target / MARKER_NAME
    if target.exists():
        _require(target.is_dir(), "Recovery target is not a directory")
        _safe_child(marker_path, target)
        if not marker_path.is_file():
            return {**result, "reason": "existing_case_untouched"}
    candidates = list(root.glob(f".{case_id}.initializing-*"))
    if not candidates:
        _require(not marker_path.exists(), "Recovery marker source is missing")
        return {**result, "reason": "no_initialization"}
    _require(len(candidates) == 1,
             f"Recovery requires one initialization; found {len(candidates)} for {case_id}")
    source = _safe_child(candidates[0], root)
    _require(source.is_dir(), "Initialization candidate is not a directory")
    epoch, hashes = _read_candidate(source, project, case_id, allow_synthetic)
    result.update(source=str(source), epoch=epoch, hashes=hashes)
    identity = {"schema_version": 1, "helper": HELPER_ID, "case_id": case_id,
                "project_root": str(project), "runs_root": str(root),
                "source": str(source), "target": str(target),
                "epoch": epoch, "hashes": hashes}
    if target.exists():
        marker = load_json(marker_path)
        _require(isinstance(marker, dict) and set(marker) == {*identity, "complete"}
                 and type(marker.get("complete")) is bool
                 and all(marker.get(key) == value for key, value in identity.items()),
                 "Recovery marker does not match the unchanged source")
        for entry in target.iterdir():
            _safe_child(entry, target)
            _require(entry.is_file() and
                     (entry.name in hashes or entry.name == MARKER_NAME or _owned_temporary(entry.name)
                      or (marker["complete"] and entry.name in ("best.pth", *EVALUATION_FILES))),
                     f"Recovery target contains an unknown file: {entry}")
        if marker["complete"]:
            _require(file_sha256(target / "config.json") == hashes["config.json"],
                     "Recovered case configuration changed")
            _, last, _, _, _, _ = read_boundary(
                target, protocol.training_source(), require_complete=False)
            validate_case_config(last["config"], allow_synthetic=allow_synthetic)
            _require(last["epoch"] >= epoch, "Recovered case moved behind its initialization")
            return {**result, "epoch": last["epoch"], "reason": "already_recovered"}
        # Inspect every existing destination before publishing any missing file.
        for name, digest in hashes.items():
            destination = _safe_child(target / name, target)
            if destination.exists():
                _require(destination.is_file() and file_sha256(destination) == digest,
                         f"Recovery refuses a different existing file: {destination}")
    else:
        target.mkdir()  # Ordinary inherited Windows ACLs; no directory rename.
        _write_marker(target, {**identity, "complete": False})
    for name, digest in hashes.items():
        source_file = _safe_child(source / name, source)
        _require(file_sha256(source_file) == digest, "Initialization changed during recovery")
        destination = _safe_child(target / name, target)
        if destination.exists():
            _require(file_sha256(destination) == digest,
                     f"Recovery refuses a different existing file: {destination}")
            continue
        temporary = _safe_child(target / f".extended-recovery-{uuid.uuid4().hex}.tmp", target)
        with source_file.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _require(file_sha256(temporary) == file_sha256(source_file) == digest,
                 "Initialization changed or a copy failed its hash check")
        _require(not destination.exists(), "Another process published the destination; retry recovery")
        os.replace(temporary, destination)
    _require(all(file_sha256(source / name) == file_sha256(target / name) == digest
                 for name, digest in hashes.items()), "Recovery source/target byte hashes differ")
    _, last, _, _, _, _ = read_boundary(target, protocol.training_source(), require_complete=False)
    validate_case_config(last["config"], allow_synthetic=allow_synthetic)
    _require(last["epoch"] == epoch and last["config"]["case_id"] == case_id,
             "Recovered target boundary differs from its source")
    _write_marker(target, {**identity, "complete": True})
    return {**result, "recovered": True, "reason": "initialization_published"}
