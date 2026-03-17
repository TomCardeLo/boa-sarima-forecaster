# Data Directory

This directory holds input data for the `sarima-bayes` forecasting pipeline.
**Real data files are excluded from version control** (see `.gitignore`).

---

## Directory layout

```
data/
├── sample_data.csv          ← Synthetic demo data (committed, safe to share)
├── input/                   ← Your real Excel files go here (git-ignored)
│   ├── sales.xlsx
│   └── representatives.xlsx
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

| Column      | Type    | Format / notes                                                   |
|-------------|---------|------------------------------------------------------------------|
| `Date`      | string  | Period in `YYYYMM` format — e.g. `"202201"` = January 2022     |
| `SKU`       | integer | Numeric product identifier                                       |
| `CS`        | float   | Demand volume (case-equivalents, units, or any numeric measure)  |
| `Country`   | string  | Market code — e.g. `"US"`, `"MX"`, `"BR"`                      |

#### Optional columns

| Column           | Type   | Description                                         |
|------------------|--------|-----------------------------------------------------|
| `Forecast group` | string | Distribution channel used for channel-level splits  |

> Additional columns present in the file are passed through unchanged.

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

### 2. `representatives.xlsx` — SKU consolidation mapping

Used **only** if you call `preprocessor.merge_representatives()`.  Maps
detailed child SKUs to a single representative parent SKU for forecasting.

#### Sheet structure

The sheet can have any name; specify it in `config.yaml` under
`data.representatives_path`.  No rows are skipped (`skip_rows=0`).

#### Required columns

| Column     | Type    | Description                                                         |
|------------|---------|---------------------------------------------------------------------|
| `Country`  | string  | Market code — must match values in `sales.xlsx`                     |
| `SKU`      | integer | Source (child) SKU identifier                                       |
| `To SKU`   | integer | Target (representative / parent) SKU identifier                     |

> Rows where `To SKU` is blank / NaN are interpreted as self-mapping
> (the SKU is its own representative and will not be consolidated).

#### Sample rows

```
Country | SKU  | To SKU
US      | 1001 | 1001
US      | 1004 | 1001   ← SKU 1004 is consolidated into SKU 1001
US      | 1005 | 1001   ← SKU 1005 is consolidated into SKU 1001
MX      | 1002 | 1002
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
