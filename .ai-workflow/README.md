# Multi-agent workflow MVP

This directory is the project-level contract shared by Codex, Cline, and Roo. It does not
choose a model. It defines role boundaries, handoffs, and executable gates so that changing a
prompt cannot masquerade as independent work.

## Runtime architecture

```text
User
  -> Orchestrator (read + delegate + gate decisions)
       -> Developer (isolated snapshot, owned production files only)
       -> Reviewer  (fresh read-only context, diff + acceptance criteria)
       -> Tester    (fresh context, tests and owned test files only)
  -> Final gate (review approved AND tests passed)
```

The existing `roo-worker-bridge` is the preferred Roo execution backend because each call starts
a separate Cline CLI process in a temporary snapshot and copies back only explicitly owned files
after a conflict check. The protocol in this directory remains valid if the backend changes.

## Roles and permissions

| Role | May read | May write | Required output | Must not do |
| --- | --- | --- | --- | --- |
| Orchestrator | Project, diff, handoffs | Workflow records only | task, assignments, gate decision | edit production code; self-approve |
| Developer | Assigned context | exact owned files | developer handoff | review or approve own work |
| Reviewer | task, diff, relevant files | nothing | findings and verdict | reuse developer context; edit files |
| Tester | task, changed code | exact owned test files only | commands, exit codes, gaps | edit production code; waive failures |
| Research/Writer | verified evidence only | assigned reports/manuscript files | claims linked to evidence | invent results or cite unverified outputs |

## Task lifecycle

1. Copy `templates/task.json` to `.ai-workflow/runs/<task-id>/task.json` and bind every placeholder.
2. Assign non-overlapping exact paths. Record a unique `actor_id` for every isolated worker run.
3. Developer returns `handoffs/developer.json`; validate it before review.
4. A fresh reviewer returns `handoffs/reviewer.json`. Any blocking finding routes back to a new
   developer attempt and invalidates downstream approval.
5. A fresh tester returns `handoffs/tester.json` with real commands and integer exit codes.
6. Run the final gate. Only a zero exit code permits completion.

```powershell
python .ai-workflow/scripts/workflow_gate.py handoff .ai-workflow/runs/<task-id>/handoffs/developer.json
python .ai-workflow/scripts/workflow_gate.py handoff .ai-workflow/runs/<task-id>/handoffs/reviewer.json
python .ai-workflow/scripts/workflow_gate.py handoff .ai-workflow/runs/<task-id>/handoffs/tester.json
python .ai-workflow/scripts/workflow_gate.py final .ai-workflow/runs/<task-id>
```

## Handoff contract

Every handoff is JSON and contains the task ID, attempt, role, unique actor ID, UTC timestamps,
owned paths, summary, evidence, and status. Role-specific fields are validated by
`workflow_gate.py` against `schemas/handoff.schema.json`.

A model response is not evidence by itself. Evidence is a repository-relative path, command with
exit code, diff fact, test report, or generated scientific artifact that the orchestrator can
inspect.

## Gates

- Review gate: reviewer actor differs from developer; reviewer wrote no files; verdict is
  `approved`; no open severity 0 or 1 findings.
- Test gate: tester actor differs from developer and reviewer; every recorded command exited 0;
  status is `passed`; gaps are explicitly listed.
- Final gate: the newest successful developer attempt is followed by both passing gates. Any
  later implementation invalidates earlier review and test evidence.

The gate checks workflow records, not scientific truth. Numerical claims still require the
research validation chain in `research/paper2agent.md`.

