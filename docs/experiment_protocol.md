# Experiment Protocol

## Scope

This protocol defines the post-TASK-13 execution order for Dynamic Heterogeneous MEC Offloading experiments.

- Order: `Smoke -> Core -> Scale`
- No standalone runner
- No YAML orchestration layer
- Use the same checkpoint selection rule across learned methods

## Stage 1: Smoke

Purpose: validate wiring and numerical stability before long runs.

Recommended setup:

```text
num_agents = 5
seeds      = 1
episodes   = 50-100
methods    = {B1, B3, B4, B6, A1, QAG}
```

Mandatory checks:

- actor observation dim = 16
- core observation dim = 14
- server info dim = 3
- no NaN / Inf in losses, rewards, or queue metrics
- QAG executes successfully
- role sigma diagnostics are logged for role-based methods

## Stage 2: Core

Purpose: main technical comparison stage.

Recommended setup:

```text
num_agents = {5, 10}
seeds      = 3
episodes   = full training budget
methods    = {B1, B2, B3, B4, B5, B6, A1, A2, A6A, A6B, QAG}
```

This is the first stage intended for publication-grade plots.

## Stage 3: Scale

Purpose: larger-agent stress test after Core is stable.

Recommended setup:

```text
num_agents = {15, 20}
seeds      = 3-5
methods    = {B1, B3, B6, A1, QAG}
```

Notes:

- `MADDPG` is optional and should be deferred until the main method is stable.

## Reporting Rules

- Report mean and dispersion across seeds.
- Keep training curves separate from final evaluation tables.
- Report queue metrics together with reward and task-cost metrics.
- Include role diagnostics for role-based methods.
- Keep the checkpoint selection rule consistent across learned methods.

## Suggested Commands

Smoke evaluation:

```bash
python -m src.evaluate --checkpoint models/checkpoint_final.pt --episodes 5 --protocol-stage smoke
python -m src.evaluate --variant-id QAG --episodes 5 --protocol-stage smoke
python -m src.visualize --output-root . --protocol-stage smoke
```

Core / Scale aggregation:

```bash
python -m src.visualize --output-root . --protocol-stage core
python -m src.visualize --output-root . --protocol-stage scale
```

Manual rerun after `paper_run.py` stage execution:

```bash
python -m src.visualize --output-root /path/to/workspace/core --protocol-stage core
python -m src.visualize --output-root /path/to/workspace/scale --protocol-stage scale --paper-only
```

When `paper_run_manifest.json` is present at the stage root, `src.visualize` reuses the
same selected summary files that the automatic runner used, so manual rerenders match the
auto-generated paper plots numerically.

Generated artifacts:

- `results/evaluation_*_summary.json`
- `results/evaluation_*_trace.jsonl`
- `results/protocol_seed_aggregation.json`
- `results/plots/*.png`
