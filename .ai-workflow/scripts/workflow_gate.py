#!/usr/bin/env python3
"""Validate role handoffs and enforce independent review/test gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import sys


ROLES = {"developer", "reviewer", "tester", "writer"}
STATUSES = {"implemented", "approved", "changes_requested", "passed", "failed", "blocked"}
REQUIRED = {
    "schema_version", "task_id", "attempt", "role", "actor_id", "started_at",
    "finished_at", "owned_paths", "summary", "evidence", "status",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def valid_time(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
        return True
    except ValueError:
        return False


def valid_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def validate_handoff(data: dict) -> list[str]:
    errors = [f"missing field: {name}" for name in sorted(REQUIRED - data.keys())]
    if errors:
        return errors
    if data["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if not isinstance(data["task_id"], str) or not data["task_id"]:
        errors.append("task_id must be a nonempty string")
    if not isinstance(data["attempt"], int) or data["attempt"] < 1:
        errors.append("attempt must be a positive integer")
    role = data["role"]
    if role not in ROLES:
        errors.append(f"unknown role: {role}")
    if not isinstance(data["actor_id"], str) or not data["actor_id"].strip():
        errors.append("actor_id must be nonempty")
    for key in ("started_at", "finished_at"):
        if not valid_time(data[key]):
            errors.append(f"{key} must be an ISO-8601 UTC timestamp ending in Z")
    if isinstance(data.get("started_at"), str) and isinstance(data.get("finished_at"), str):
        if valid_time(data["started_at"]) and valid_time(data["finished_at"]) and data["finished_at"] < data["started_at"]:
            errors.append("finished_at precedes started_at")
    owned = data["owned_paths"]
    if not isinstance(owned, list) or any(not valid_path(item) for item in owned):
        errors.append("owned_paths must contain safe repository-relative POSIX paths")
    elif len(owned) != len(set(owned)):
        errors.append("owned_paths contains duplicates")
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        errors.append("summary must be nonempty")
    if not isinstance(data["evidence"], list) or any(not isinstance(x, str) for x in data["evidence"]):
        errors.append("evidence must be a string array")
    if data["status"] not in STATUSES:
        errors.append(f"unknown status: {data['status']}")

    if role == "developer":
        changed = data.get("changed_files")
        if data["status"] == "implemented" and (not isinstance(changed, list) or not changed):
            errors.append("implemented developer handoff requires changed_files")
        elif isinstance(changed, list):
            if any(not valid_path(item) for item in changed):
                errors.append("changed_files contains an unsafe path")
            if set(changed) - set(owned):
                errors.append("developer changed_files exceed owned_paths")
    elif role == "reviewer":
        if owned:
            errors.append("reviewer must be read-only (owned_paths must be empty)")
        verdict = data.get("verdict")
        if verdict not in {"approved", "changes_requested"}:
            errors.append("reviewer verdict is required")
        findings = data.get("findings")
        if not isinstance(findings, list):
            errors.append("reviewer findings must be an array")
        elif verdict == "approved" and any(
            isinstance(item, dict) and item.get("severity") in {0, 1} and not item.get("resolved", False)
            for item in findings
        ):
            errors.append("reviewer cannot approve with unresolved severity 0/1 findings")
        if data["status"] != verdict:
            errors.append("reviewer status must match verdict")
    elif role == "tester":
        commands = data.get("commands")
        if not isinstance(commands, list) or not commands:
            errors.append("tester commands must be a nonempty array")
        elif any(not isinstance(item, dict) or not isinstance(item.get("exit_code"), int) for item in commands):
            errors.append("each tester command requires an integer exit_code")
        if not isinstance(data.get("gaps"), list):
            errors.append("tester gaps must be an array")
        if data["status"] == "passed" and isinstance(commands, list) and any(item.get("exit_code") != 0 for item in commands if isinstance(item, dict)):
            errors.append("tester cannot pass with a nonzero exit code")
    return errors


def newest(paths: list[Path], role: str) -> tuple[Path, dict]:
    candidates = []
    for path in paths:
        data = load_json(path)
        if data.get("role") == role:
            candidates.append((int(data.get("attempt", 0)), data.get("finished_at", ""), path, data))
    if not candidates:
        raise ValueError(f"missing {role} handoff")
    _, _, path, data = max(candidates, key=lambda item: (item[0], item[1]))
    return path, data


def final_gate(run_dir: Path) -> list[str]:
    handoff_dir = run_dir / "handoffs"
    paths = sorted(handoff_dir.glob("*.json"))
    errors: list[str] = []
    selected = {}
    for role in ("developer", "reviewer", "tester"):
        try:
            path, data = newest(paths, role)
            selected[role] = data
            errors.extend(f"{path}: {message}" for message in validate_handoff(data))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors or len(selected) != 3:
        return errors
    developer, reviewer, tester = selected["developer"], selected["reviewer"], selected["tester"]
    if len({developer["actor_id"], reviewer["actor_id"], tester["actor_id"]}) != 3:
        errors.append("developer, reviewer, and tester must use distinct actor_id values")
    if not (developer["finished_at"] <= reviewer["started_at"] and developer["finished_at"] <= tester["started_at"]):
        errors.append("review and test must start after the selected developer finishes")
    if developer["status"] != "implemented":
        errors.append("latest developer handoff is not implemented")
    if reviewer.get("verdict") != "approved" or reviewer["status"] != "approved":
        errors.append("review gate did not approve")
    if tester["status"] != "passed":
        errors.append("test gate did not pass")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    handoff = sub.add_parser("handoff")
    handoff.add_argument("path", type=Path)
    final = sub.add_parser("final")
    final.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        errors = validate_handoff(load_json(args.path)) if args.command == "handoff" else final_gate(args.run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors = [str(exc)]
    result = {"success": not errors, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())

