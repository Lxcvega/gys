# Project agent workflow

Use `.ai-workflow/README.md` as the source of truth for multi-agent work in this repository.

- The orchestrator coordinates work but does not implement production changes.
- A developer receives exact file ownership and works in an isolated context.
- A reviewer is read-only and must use a fresh context after implementation.
- A tester may change only explicitly assigned test files and must use a fresh context.
- A task is complete only after the review and test gates pass through
  `.ai-workflow/scripts/workflow_gate.py`.
- Preserve unrelated and uncommitted work. Never clean, reset, or overwrite it.
- For paper or research-code conversion, use the installed `paper2agent` skill and follow
  `.ai-workflow/research/paper2agent.md`.

