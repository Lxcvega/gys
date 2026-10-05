# Paper2Agent research chain

The installed official `paper2agent` skill is the conversion entry point. Do not copy paper
algorithms into prompts or invent wrappers. Paper-derived tools must bind to real source code and
pass the skill's independent verification.

## Minimum runnable chain

```text
papers / supplements / code / data
  -> Paper2Skill (searchable paper skill)
  -> Paper2MCP (tested wrappers around upstream code)
  -> local MCP registration
  -> research orchestrator
  -> isolated Python experiment
  -> baseline -> ablation -> statistics
  -> figures/tables
  -> writing agent using verified evidence only
```

## Project layout

Keep generated environments and Paper2Agent evidence outside the upstream scientific source tree.

```text
research/
  sources/<study>/              # source manifest; originals may remain external
  paper-agents/<study>/         # Paper2Agent working project
  experiments/<study>/
    manifest.json
    configs/
    runs/
    tables/
    figures/
    reports/
  writing/<study>/
```

## Conversion gate

Required inputs are a paper/local source or a code repository, an output directory, and the
scientific tasks to expose. For combined paper and code input, request both Paper2Skill and
Paper2MCP output. Paper2MCP requires real isolated implementer and fresh verifier contexts.

Example request:

```text
Use the paper2agent skill to convert <paper files> and <repository> into a reviewed paper skill
and tested MCP server under research/paper-agents/<study>. Focus on <scientific tasks>.
```

Do not register the generated MCP until all of these pass:

- source identity and selected tools recorded;
- upstream examples executed with retained reference results;
- implementer and verifier actor IDs are distinct;
- MCP inventory and saved acceptance calls pass in project and fresh runtimes;
- `USAGE.md` and extracted ZIP validation pass;
- credentials remain in environment/secret storage, never project JSON.

## Experiment gate

Each run records source commit/tree hash, environment lock, seed, command, configuration, input
hashes, output hashes, metrics, and hardware notes. The default sequence is:

1. Reproduce one upstream reference result within a declared tolerance.
2. Run a project baseline on the same split and metrics.
3. Run one-factor ablations with all uncontrolled settings fixed.
4. Repeat across declared seeds/folds and report uncertainty or statistical tests appropriate to
   the metric; do not select only favorable runs.
5. Generate figures and tables from saved machine-readable results.
6. Permit writing only from the verified manifest, tables, figures, and limitations report.

Paper2Agent verifies the paper-derived MCP; it does not by itself validate this project's new
scientific conclusions. The experiment and writing gates remain mandatory.

