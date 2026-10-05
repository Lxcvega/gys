import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "workflow_gate.py"
SPEC = importlib.util.spec_from_file_location("workflow_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(gate)


def record(role, actor, start, finish, status, **extra):
    value = {
        "schema_version": 1,
        "task_id": "demo-task",
        "attempt": 1,
        "role": role,
        "actor_id": actor,
        "started_at": start,
        "finished_at": finish,
        "owned_paths": [] if role == "reviewer" else ["tests/test_demo.py" if role == "tester" else "src/demo.py"],
        "summary": "evidence-backed handoff",
        "evidence": ["artifact"],
        "status": status,
    }
    value.update(extra)
    return value


class WorkflowGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "handoffs").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        (self.run / "handoffs" / name).write_text(json.dumps(value), encoding="utf-8")

    def valid_records(self):
        self.write("developer.json", record("developer", "dev-1", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "implemented", changed_files=["src/demo.py"]))
        self.write("reviewer.json", record("reviewer", "review-1", "2026-01-01T00:02:00Z", "2026-01-01T00:03:00Z", "approved", verdict="approved", findings=[]))
        self.write("tester.json", record("tester", "test-1", "2026-01-01T00:02:00Z", "2026-01-01T00:04:00Z", "passed", commands=[{"command": "pytest", "exit_code": 0}], gaps=[]))

    def test_final_gate_passes_independent_roles(self):
        self.valid_records()
        self.assertEqual([], gate.final_gate(self.run))

    def test_same_actor_fails_independence(self):
        self.valid_records()
        path = self.run / "handoffs" / "reviewer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["actor_id"] = "dev-1"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertTrue(any("distinct actor_id" in item for item in gate.final_gate(self.run)))

    def test_failed_test_command_cannot_pass(self):
        value = record("tester", "test-1", "2026-01-01T00:02:00Z", "2026-01-01T00:04:00Z", "passed", commands=[{"command": "pytest", "exit_code": 1}], gaps=[])
        self.assertTrue(any("nonzero" in item for item in gate.validate_handoff(value)))


if __name__ == "__main__":
    unittest.main()

