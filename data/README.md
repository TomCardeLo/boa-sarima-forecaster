# Data Directory

This directory holds input data for the `sarima-bayes` forecasting pipeline.
**Real data files are excluded from version control** (see `.gitignore`).

---

## Directory layout

```
data/
├── sample_data.csv          ← Synthetic demo data (committed, safe to share)
├── input/                   ← Your real Excel files go here (git-ignored)
│   └── sales.xlsx
└── output/                  ← Forecast results written here (git-ignored)
```

---

## Input file specifications

### 1. `sales.xlsx` — Main sales history

This is the primary input consumed by `data_loader.load_data()`.

#### Sheet structure

| Row   | Content                          |
|-------|----------------------------------|
| 0     | Blank or metadata (skipped)      |
| 1     | Blank or metadata (skipped)      |
| 2     | **Column headers**               |
| 3+    | Data rows                        |

> The two skipped rows (`skip_rows=2`) correspond to a common export format
> where the first rows contain report metadata.  If your file has no extra
> header rows, set `skip_rows: 0` in `config.yaml`.

#### Required columns

| Column | Type   | Format / notes                                                      |
|--------|--------|---------------------------------------------------------------------|
| `Date` | string | Period in `YYYYMM` format — e.g. `"202201"` = January 2022        |
| `CS`   | float  | Target variable (case-equivalents, units, revenue, or any numeric measure) |

#### Optional columns

| Column           | Type    | Default | Description                                         |
|------------------|---------|---------|-----------------------------------------------------|
| `SKU`            | integer | `1`     | Series identifier — omit for single-series use      |
| `Country`        | string  | `"_"`   | Market code — e.g. `"US"`, `"MX"`, `"BR"`          |
| `Forecast group` | string  | —       | Distribution channel used for channel-level splits  |

> `SKU` and `Country` are **auto-injected** with their defaults if not present.
> Additional columns are passed through unchanged.

#### Sample rows (as they would appear in Excel)

```
(row 0 — blank)
(row 1 — blank)
Date     | SKU  | CS    | Country
202201   | 1001 | 148.0 | US
202202   | 1001 | 142.0 | US
202201   | 1002 | 195.0 | US
...
```

---

## `sample_data.csv` — Synthetic demo data

This file contains **synthetic** data generated to illustrate the expected
format.  It is safe to commit and share.

- **Period**: January 2022 – December 2024 (36 months)
- **SKUs**: 1001, 1002, 1003
- **Countries**: US, MX
- **Series**: 6 independent time series (3 SKUs × 2 countries)
- **Pattern**: mild upward trend + Q4 seasonality peak + random noise

The `notebooks/demo.ipynb` notebook loads this CSV directly and also shows
how to generate additional synthetic data programmatically.

---

## Date format reference

| Raw cell value | Parsed date        | Notes                   |
|----------------|--------------------|-------------------------|
| `202201`       | 2022-01-01         | January 2022            |
| `202212`       | 2022-12-01         | December 2022           |
| `202306`       | 2023-06-01         | June 2023               |

All dates are normalised to **month-start** (`MS`) frequency before any
preprocessing or model fitting step.
