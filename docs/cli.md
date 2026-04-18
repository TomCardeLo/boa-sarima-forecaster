# CLI Reference

`boa-forecaster` ships with a Click-based command-line entry point registered
as the console script `boa-forecaster` (also reachable as
`python -m boa_forecaster`).

## Installation

```bash
pip install -e .
```

After installation, the `boa-forecaster` command is available on `PATH`.

## Subcommands

### `run`

Optimise the active model declared in `config.yaml` and emit a forecast.

```bash
boa-forecaster run --config config.yaml --output out/
```

Outputs, written to `--output`:

| File           | Contents                                                  |
|----------------|-----------------------------------------------------------|
| `forecast.csv` | `date,forecast` — point forecast over the optimiser horizon |
| `params.json`  | `best_params` + `best_score` from the TPE study           |
| `plot.png`     | Matplotlib plot (only when `matplotlib` is installed)     |

Flags:

- `--n-trials INT` — override `optimization.n_calls`.

### `validate`

Run walk-forward (expanding-window) cross-validation for the active model.

```bash
boa-forecaster validate --config config.yaml --output out/ \
    --n-folds 3 --test-size 12 --min-train-size 24
```

Writes `folds.csv` (one row per fold, with sMAPE/RMSLE) and `metrics.csv`
(aggregated mean/std), plus the same forecast artefacts as `run`.

### `compare`

Optimise every `models.*` entry with `enabled: true` plus zero-budget
baselines (Seasonal-Naïve, ETS) and rank them on the same walk-forward
folds.

```bash
boa-forecaster compare --config config.yaml --output out/
```

Writes `metrics.csv` with one row per model and columns
`model, sMAPE_mean, sMAPE_std, RMSLE_mean, RMSLE_std`.

Use `--no-baselines` to skip Seasonal-Naïve and ETS.

## Configuration

All subcommands consume a YAML file validated by
`boa_forecaster.config_schema.BoaConfig`.  Copy
[`config.example.yaml`](../config.example.yaml) and edit.  Typos in field
names are tolerated in v2.x (warns via Pydantic `extra="allow"`) but will
raise `ValidationError` once the v3.0 strict-mode switch flips.
