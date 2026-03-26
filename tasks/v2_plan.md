# BOA-Forecaster v2.0 — Plan de Implementación

> **Objetivo**: Evolucionar el repositorio de un optimizador Bayesiano específico para SARIMA
> a un framework de optimización Bayesiana TPE agnóstico al modelo, con soporte nativo para
> modelos de series temporales (SARIMA, ETS) y modelos ML (Random Forest, XGBoost, LightGBM).

---

## Tabla de contenidos

1. [Visión y principios de diseño](#1-visión-y-principios-de-diseño)
2. [Arquitectura target v2.0](#2-arquitectura-target-v20)
3. [Contratos de interfaz](#3-contratos-de-interfaz)
4. [Estrategia de compatibilidad hacia atrás](#4-estrategia-de-compatibilidad-hacia-atrás)
5. [Plan de fases y tareas](#5-plan-de-fases-y-tareas)
6. [Diseño detallado por módulo](#6-diseño-detallado-por-módulo)
7. [Esquema de configuración YAML v2.0](#7-esquema-de-configuración-yaml-v20)
8. [Estrategia de testing](#8-estrategia-de-testing)
9. [Gestión de dependencias opcionales](#9-gestión-de-dependencias-opcionales)
10. [Checklist de Definition of Done por fase](#10-checklist-de-definition-of-done-por-fase)

---

## 1. Visión y principios de diseño

### Propuesta de valor v2.0

> "Un único framework de optimización Bayesiana TPE para ajustar hiperparámetros de
> cualquier modelo de forecasting — desde SARIMA hasta LightGBM — con la misma
> pipeline de validación, métricas y comparación de baselines."

### Principios de diseño

| Principio | Implicación concreta |
|-----------|----------------------|
| **Plugin-first** | Cada modelo es un `ModelSpec` registrable; añadir un modelo nuevo no toca el núcleo |
| **Backwards-compatible** | `optimize_arima()` sigue funcionando en v2.0, con deprecation warning |
| **Temporal integrity** | Feature engineering para ML nunca filtra datos futuros en CV |
| **Soft failures** | El optimizador no se rompe si un modelo falla; devuelve penalización y continúa |
| **Optional deps** | XGBoost/LightGBM son extras opcionales; el core no los requiere |
| **Minimal interface** | El contrato `ModelSpec` es delgado — solo 4 métodos abstractos |
| **Config-driven** | Cualquier search space se puede definir en `config.yaml` sin tocar código |

### Qué NO es v2.0

- No es un AutoML general (no hace selección automática de modelo)
- No soporta datos multivariados (fuera de scope por ahora)
- No hace hiperparameter optimization en tiempo real / online
- No reemplaza herramientas como Optuna Dashboards (aunque es compatible)

---

## 2. Arquitectura target v2.0

### Estructura de paquete

```
src/
├── boa_forecaster/              ← NUEVO paquete principal v2.0
│   ├── __init__.py              ✅ CREADO (Fase 0) — API pública v2.0-dev
│   ├── config.py                ✅ CREADO (Fase 0) — + OPTIMIZER_PENALTY añadido
│   ├── optimizer.py             ← Motor TPE genérico (Fase 1)
│   ├── features.py              ← NUEVO: Feature engineering para ML (Fase 1)
│   ├── metrics.py               ✅ CREADO (Fase 0)
│   ├── validation.py            ✅ CREADO (Fase 0)
│   ├── preprocessor.py          ✅ CREADO (Fase 0)
│   ├── data_loader.py           ✅ CREADO (Fase 0)
│   ├── standardization.py       ✅ CREADO (Fase 0)
│   ├── benchmarks.py            ← Ampliado para soportar modelos ML (Fase 5)
│   └── models/
│       ├── __init__.py          ✅ CREADO (Fase 0) — placeholder
│       ├── base.py              ← NUEVO: ModelSpec Protocol + tipos (Fase 1)
│       ├── sarima.py            ← NUEVO: SARIMASpec (Fase 1)
│       ├── random_forest.py     ← NUEVO: RandomForestSpec (Fase 2)
│       ├── xgboost.py           ← NUEVO: XGBoostSpec (optional dep) (Fase 3)
│       └── lightgbm.py          ← NUEVO: LightGBMSpec (optional dep) (Fase 4)
│
└── sarima_bayes/                ← EXISTENTE — mantener como alias deprecado (Fase 6)
    └── __init__.py              ← Reexporta todo desde boa_forecaster + DeprecationWarning
```

### Diagrama de flujo v2.0

```
Excel / DataFrame
      │
      ▼
data_loader.py ──── preprocessor.py ──── standardization.py
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   ModelSpec Registry   │
                                    │  sarima / rf / xgb /  │
                                    │       lgbm / ...       │
                                    └───────────┬───────────┘
                                                │
                        ┌───────────────────────▼───────────────────────┐
                        │                 optimizer.py                    │
                        │           optimize_model(series,                │
                        │              model_spec, n_calls)               │
                        │                                                 │
                        │  ┌──────────────────────────────────────────┐  │
                        │  │           Optuna TPE Engine               │  │
                        │  │  - multivariate=True, seed=42             │  │
                        │  │  - Warm-start desde model_spec            │  │
                        │  │  - suggest_params() → model_spec          │  │
                        │  │  - evaluate() → metric score              │  │
                        │  └──────────────────────────────────────────┘  │
                        └───────────────────────┬───────────────────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │         features.py                │
                              │   (solo si model_spec.needs_features)│
                              │   lags + rolling + calendar        │
                              └─────────────────┬─────────────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │           validation.py            │
                              │   walk_forward_validation          │
                              │   (expanding window, sin cambios)  │
                              └─────────────────┬─────────────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │           benchmarks.py            │
                              │   run_benchmark_comparison         │
                              │   (SARIMA + RF + XGB + LGBM +      │
                              │    Seasonal Naïve + ETS)           │
                              └────────────────────────────────────┘
```

---

## 3. Contratos de interfaz

### 3.1 `ModelSpec` Protocol (`models/base.py`)

Este es el contrato central. Cualquier modelo que lo implemente puede ser optimizado.

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
import pandas as pd

# ── Definición de parámetros del search space ────────────────────────────────

@dataclass
class IntParam:
    low: int
    high: int
    step: int = 1
    log: bool = False

@dataclass
class FloatParam:
    low: float
    high: float
    log: bool = False

@dataclass
class CategoricalParam:
    choices: list  # puede mezclar tipos: ["sqrt", 0.5, "log2"]

SearchSpaceParam = IntParam | FloatParam | CategoricalParam

# ── Resultado del optimizador ────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    best_params: dict
    best_score: float
    n_trials: int
    model_name: str

# ── Protocolo ModelSpec ──────────────────────────────────────────────────────

@runtime_checkable
class ModelSpec(Protocol):
    """
    Contrato que todo modelo debe implementar para ser optimizable con TPE.

    Implementar este protocolo (structural subtyping) es suficiente —
    no se requiere herencia explícita de ninguna clase base.
    """

    name: str
    """Identificador único del modelo en el registry."""

    needs_features: bool
    """
    True  → el modelo requiere FeatureEngineer (ML tabular).
    False → el modelo consume pd.Series directamente (modelos AR, ETS).
    """

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """
        Define el espacio de búsqueda de hiperparámetros.

        Cada clave es un nombre de parámetro; el valor es un SearchSpaceParam
        que Optuna usará para sugerir valores.

        Ejemplo:
            {
                "n_estimators": IntParam(50, 500, log=True),
                "max_depth":    IntParam(2, 20),
                "max_features": CategoricalParam(["sqrt", "log2", 0.5]),
            }
        """
        ...

    @property
    def warm_starts(self) -> list[dict]:
        """
        Lista de configuraciones de parámetros para pre-sembrar en Optuna
        antes de que TPE tome el control.

        Propósito: guiar la búsqueda inicial con combinaciones conocidas.

        Ejemplo:
            [
                {"n_estimators": 100, "max_depth": 5, "max_features": "sqrt"},
                {"n_estimators": 200, "max_depth": 10, "max_features": "log2"},
            ]
        """
        ...

    def suggest_params(self, trial) -> dict:
        """
        Usa trial.suggest_int / suggest_float / suggest_categorical para
        muestrear un punto del search_space.

        Recibe un optuna.Trial; devuelve dict de parámetros.
        """
        ...

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config=None,
    ) -> float:
        """
        Ajusta el modelo con `params` sobre `series` y devuelve el score
        de la función objetivo `metric_fn(y_true, y_pred) -> float`.

        - Si el modelo falla, devuelve OPTIMIZER_PENALTY (1e6).
        - `feature_config` solo es relevante si needs_features=True.
        - Internamente debe usar walk_forward_validation o un fold único
          para estimar la generalización.
        """
        ...

    def build_forecaster(self, params: dict, feature_config=None):
        """
        Dado un conjunto de hiperparámetros óptimos, devuelve un callable:

            forecaster(train: pd.Series) -> pd.Series

        donde el pd.Series devuelto contiene las predicciones futuras con
        DatetimeIndex correcto y longitud = forecast_horizon.

        Este callable es compatible con walk_forward_validation y benchmarks.
        """
        ...
```

### 3.2 `FeatureConfig` y `FeatureEngineer` (`features.py`)

```python
@dataclass
class FeatureConfig:
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 12])
    include_calendar: bool = True      # mes, trimestre (encoding cíclico sin/cos)
    include_trend: bool = True         # índice temporal normalizado [0,1]
    include_expanding: bool = False    # mean/std expanding window
    target_col: str = "y"

class FeatureEngineer:
    """
    Transforma pd.Series → pd.DataFrame de features para modelos ML.
    Respeta la causalidad temporal: nunca usa datos futuros.

    Uso típico en CV:
        fe = FeatureEngineer(config)
        X_train, y_train = fe.fit_transform(train_series)
        X_test = fe.transform(test_series)  # usa stats de train
    """
    def fit_transform(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]: ...
    def transform(self, series: pd.Series) -> pd.DataFrame: ...
    def get_feature_names(self) -> list[str]: ...
```

### 3.3 `optimize_model` genérico (`optimizer.py`)

```python
def optimize_model(
    series: pd.Series,
    model_spec: ModelSpec,
    n_calls: int = 50,
    n_jobs: int = 1,
    metric_components: list[dict] | None = None,
    feature_config: FeatureConfig | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Motor TPE genérico. Optimiza cualquier ModelSpec.

    Parámetros
    ----------
    series          : Serie temporal de entrenamiento con DatetimeIndex.
    model_spec      : Instancia de cualquier clase que implemente ModelSpec.
    n_calls         : Número de trials de Optuna (incluye warm_starts).
    n_jobs          : Paralelismo (< 1 → todos los CPUs disponibles).
    metric_components: Lista de {metric, weight} para la función objetivo.
                      None → usa sMAPE(0.7) + RMSLE(0.3).
    feature_config  : Configuración de features para modelos ML.
                      Ignorada si model_spec.needs_features=False.
    seed            : Semilla del TPESampler para reproducibilidad.
    verbose         : Si True, activa logging de Optuna.

    Retorna
    -------
    OptimizationResult con best_params, best_score, n_trials, model_name.
    """
```

### 3.4 API pública `__init__.py` v2.0

```python
# Motor principal
from .optimizer import optimize_model
from .optimizer import optimize_arima  # deprecated alias → optimize_model con SARIMASpec

# Modelos
from .models import MODEL_REGISTRY, register_model, get_model_spec
from .models.base import ModelSpec, FeatureConfig, OptimizationResult
from .models.base import IntParam, FloatParam, CategoricalParam
from .models.sarima import SARIMASpec
from .models.random_forest import RandomForestSpec
from .models.xgboost import XGBoostSpec     # raises ImportError si no instalado
from .models.lightgbm import LightGBMSpec   # raises ImportError si no instalado

# Feature engineering
from .features import FeatureEngineer, FeatureConfig

# Métricas (sin cambios)
from .metrics import (
    smape, rmsle, mae, rmse, mape,
    combined_metric, build_combined_metric, METRIC_REGISTRY,
)

# Validación (sin cambios)
from .validation import walk_forward_validation, validate_by_group

# Forecasting SARIMA (sin cambios, path interno actualizado)
from .models.sarima import pred_arima, forecast_arima

# Benchmarks (ampliado)
from .benchmarks import run_benchmark_comparison, summary_table
```

---

## 4. Estrategia de compatibilidad hacia atrás

### Matriz de compatibilidad

| Símbolo v1.x | Estado v2.0 | Acción |
|---|---|---|
| `from sarima_bayes import optimize_arima` | ✅ deprecado | Reexporta + `DeprecationWarning` |
| `from sarima_bayes import forecast_arima` | ✅ deprecado | Reexporta sin warning (API pública estable) |
| `from sarima_bayes import smape, rmsle, ...` | ✅ deprecado | Reexporta sin warning |
| `from sarima_bayes import walk_forward_validation` | ✅ deprecado | Reexporta sin warning |
| `from sarima_bayes import run_benchmark_comparison` | ✅ deprecado | Reexporta con nota de cambio |
| `optimize_arima(series, p_range, ...)` | ✅ funciona | Wrapper que construye SARIMASpec internamente |
| `from boa_forecaster import optimize_model` | ✅ nuevo primario | — |

### Implementación del alias deprecado (`sarima_bayes/__init__.py`)

```python
import warnings
warnings.warn(
    "El paquete 'sarima_bayes' ha sido renombrado a 'boa_forecaster'. "
    "Por favor actualiza tus imports. 'sarima_bayes' será eliminado en v3.0.",
    DeprecationWarning,
    stacklevel=2,
)
from boa_forecaster import *  # noqa: F401, F403
```

---

## 5. Plan de fases y tareas

### Resumen de fases

| Fase | Nombre | Scope | Estado |
|------|--------|-------|--------|
| **0** | Preparación | Estructura de directorios, pyproject.toml | ✅ **COMPLETADO** (2026-03-26) |
| **1** | Core Infrastructure | ModelSpec, optimizer genérico, FeatureEngineer, SARIMASpec | ✅ **COMPLETADO** (2026-03-26) |
| **2** | Random Forest | RandomForestSpec + tests + config | ✅ **COMPLETADO** (2026-03-26) |
| **3** | XGBoost | XGBoostSpec + tests + config | ✅ **COMPLETADO** (2026-03-26) |
| **4** | LightGBM | LightGBMSpec + tests + config | ✅ **COMPLETADO** (2026-03-26) |
| **5** | Benchmarks & comparación | run_benchmark_comparison ampliado | ✅ **COMPLETADO** (2026-03-26) |
| **6** | Package rename + API | boa_forecaster principal, deprecation sarima_bayes | ✅ **COMPLETADO** (2026-03-26) |
| **7** | Documentación & CI | README, CLAUDE.md, pyproject.toml extras, CI matrix | ⬜ Pendiente |

---

### FASE 0 — Preparación ✅ COMPLETADO (2026-03-26)

**Objetivo**: Crear la estructura de directorios y actualizar dependencias sin romper nada.

**Resultado**: 96 tests pasando, 0 fallos. `from boa_forecaster import smape` funciona.

#### Tarea 0.1: Crear estructura de directorios ✅
- [x] Crear `src/boa_forecaster/` con `__init__.py`
- [x] Crear `src/boa_forecaster/models/` con `__init__.py` (placeholder)
- [x] Crear `tests/integration/__init__.py`

#### Tarea 0.2: Actualizar `pyproject.toml` ✅
- [x] Extras opcionales añadidos: `sklearn`, `xgboost`, `lightgbm`, `ml`, `all`
- [x] Markers pytest añadidos: `requires_sklearn`, `requires_xgboost`, `requires_lightgbm`, `integration`
- [x] Todas las dependencias existentes mantenidas sin cambios

> **Nota**: `scikit-learn` se dejó como extra opcional (`sklearn`) en lugar de core.
> Se moverá a core cuando se implemente `RandomForestSpec` en Fase 2.

#### Tarea 0.3: Copiar módulos invariantes ✅
- [x] `config.py` — copia con `OPTIMIZER_PENALTY = 1e6` añadido (útil desde Fase 1)
- [x] `metrics.py` — copia exacta
- [x] `standardization.py` — copia exacta
- [x] `preprocessor.py` — copia exacta (sin imports internos de `sarima_bayes`)
- [x] `data_loader.py` — `from sarima_bayes.config` → `from boa_forecaster.config`
- [x] `validation.py` — `from sarima_bayes.metrics` → `from boa_forecaster.metrics`
- [x] `sarima_bayes/` intacto — tests existentes pasan al 100%

**Criterio de done**: ✅ `pytest tests/ -v` → 96 passed, 2 skipped, 0 failed.

---

### FASE 1 — Core Infrastructure ✅ COMPLETADO (2026-03-26)

**Objetivo**: El motor TPE genérico existe y funciona para SARIMA exactamente igual que antes.

**Resultado**: 175 tests pasando (acumulado), 0 fallos. `optimize_model(series, SARIMASpec())` produce resultados idénticos a `optimize_arima`.

#### Tarea 1.1: `models/base.py` — Tipos y Protocol ✅
- [x] Implementar `IntParam`, `FloatParam`, `CategoricalParam` como dataclasses
- [x] Definir `SearchSpaceParam = IntParam | FloatParam | CategoricalParam` (Union type)
- [x] Implementar `OptimizationResult` dataclass con campos: `best_params`, `best_score`, `n_trials`, `model_name`
- [x] Implementar `ModelSpec` como `typing.Protocol` con `@runtime_checkable`
- [x] Métodos del protocolo: `name`, `needs_features`, `search_space`, `warm_starts`, `suggest_params(trial)`, `evaluate(series, params, metric_fn, feature_config)`, `build_forecaster(params, feature_config)`
- [x] Implementar helper `suggest_from_space(trial, search_space: dict) -> dict` — función libre que traduce un `dict[str, SearchSpaceParam]` a llamadas Optuna
- [x] Tests en `tests/unit/test_base.py`: verificar tipos, suggest_from_space con los 3 tipos de param

#### Tarea 1.2: `features.py` — Feature Engineering ✅
- [x] Implementar `FeatureConfig` dataclass con defaults: `lag_periods=[1,2,3,6,12]`, `rolling_windows=[3,6,12]`, `include_calendar=True`, `include_trend=True`, `include_expanding=False`
- [x] Implementar clase `FeatureEngineer`:
  - `__init__(self, config: FeatureConfig)`
  - `fit_transform(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]`
  - `transform(series: pd.Series) -> pd.DataFrame`
  - `get_feature_names(self) -> list[str]`
- [x] Temporal integrity check: test que `fit_transform` no usa datos de índices > len(train)
- [x] Tests en `tests/unit/test_features.py`

#### Tarea 1.3: `models/sarima.py` — SARIMASpec ✅
- [x] Crear clase `SARIMASpec` que implemente `ModelSpec`
- [x] `name = "sarima"`, `needs_features = False`
- [x] `search_space property` devuelve `IntParam` para cada orden (p, d, q, P, D, Q)
- [x] `warm_starts property` devuelve 2 configuraciones de inicio
- [x] `suggest_params(trial)` con constraints `p+q<=4`, `P+Q<=3`
- [x] `evaluate(series, params, metric_fn, feature_config=None)` con soft-penalty `1e6`
- [x] `build_forecaster(params, feature_config=None)`
- [x] Mantener `pred_arima` y `forecast_arima` como funciones libres (compatibilidad)
- [x] Tests en `tests/unit/test_sarima_spec.py`

#### Tarea 1.4: `optimizer.py` — Motor TPE genérico ✅
- [x] Implementar `optimize_model(series, model_spec, n_calls, n_jobs, metric_components, feature_config, seed, verbose) -> OptimizationResult`
- [x] Implementar `optimize_arima(...)` como wrapper deprecado que devuelve `(dict, float)`
- [x] Tests en `tests/unit/test_optimizer_generic.py`

**Criterio de done fase 1**: ✅ Tests al 100%. `optimize_arima(series, ...)` produce resultados idénticos a v1.x.

---

### FASE 2 — Random Forest ✅ COMPLETADO (2026-03-26)

**Objetivo**: Primer modelo ML con feature engineering integrado y TPE sobre scikit-learn.

**Resultado**: 212 tests pasando (acumulado), 0 fallos. `optimize_model(series, RandomForestSpec())` funciona end-to-end. `scikit-learn` promovido a dependencia core.

#### Tarea 2.1: `models/random_forest.py` — RandomForestSpec ✅
- [x] Import guard para `sklearn.ensemble.RandomForestRegressor` (`HAS_SKLEARN`)
- [x] `__init__(self, feature_config=None, forecast_horizon=12)` — lanza `ImportError` si no hay sklearn
- [x] `name = "random_forest"`, `needs_features = True`
- [x] `search_space`: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- [x] `warm_starts`: 2 entradas con keys completas
- [x] `suggest_params(trial)`: delega a `suggest_from_space`
- [x] `evaluate(series, params, metric_fn, feature_config=None)`: 3-fold expanding-window CV inline, FeatureEngineer fresco por fold, soft-penalty
- [x] `build_forecaster(params, feature_config=None)`: closure con RF + recursive forecast
- [x] `_recursive_forecast(model, fe, train, horizon) -> pd.Series`: appends predictions step-by-step
- [x] 37 tests en `tests/unit/test_random_forest.py` — todos pasan

#### Tarea 2.2: Registrar RF en MODEL_REGISTRY ✅
- [x] `register_model("random_forest", RandomForestSpec)` en `models/__init__.py`
- [x] `RandomForestSpec` exportado en `__init__.py` y `__all__`

#### Tarea 2.3: Config YAML para RF ✅
- [x] Sección `models:` completa añadida a `config.example.yaml` (RF + XGBoost + LightGBM para Fases 3/4)
- [x] `RF_DEFAULT_N_ESTIMATORS=100`, `RF_DEFAULT_MAX_DEPTH=10` añadidos a `config.py`
- [x] `scikit-learn>=1.3` movido de extras opcionales a `dependencies` core en `pyproject.toml`

**Criterio de done fase 2**: ✅ 212 passed, 0 failed. `from boa_forecaster import RandomForestSpec` funciona. `get_model_spec("random_forest")` registrado.

---

### FASE 3 — XGBoost ✅ COMPLETADO (2026-03-26)

**Objetivo**: Gradiente boosting con hiperparámetros complejos y early stopping.

#### Tarea 3.1: `models/xgboost.py` — XGBoostSpec ✅

- [x] Import guard para `xgboost`
- [x] `name = "xgboost"`, `needs_features = True`
- [x] `__init__(self, feature_config, forecast_horizon=12, early_stopping_rounds=20)`
- [ ] `search_space`:
  ```python
  {
      "n_estimators":      IntParam(50, 1000, log=True),
      "max_depth":         IntParam(2, 10),
      "learning_rate":     FloatParam(0.005, 0.3, log=True),
      "subsample":         FloatParam(0.5, 1.0),
      "colsample_bytree":  FloatParam(0.5, 1.0),
      "min_child_weight":  IntParam(1, 20),
      "reg_alpha":         FloatParam(1e-8, 10.0, log=True),
      "reg_lambda":        FloatParam(1e-8, 10.0, log=True),
      "gamma":             FloatParam(0.0, 5.0),
  }
  ```
- [x] `warm_starts`: (nota: `reg_alpha: 1e-8` en lugar de `0.0` para respetar floor del log-scale)
  ```python
  [
      {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
       "reg_alpha": 1e-8, "reg_lambda": 1.0, "gamma": 0.0},
      {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
       "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,
       "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1},
  ]
  ```
- [x] `evaluate`: walk-forward 3 folds con FeatureEngineer + split interno 20% val para early stopping
- [x] `build_forecaster`: `xgb.XGBRegressor(**params)` con `recursive_forecast` (sin early stopping)
- [x] 53 tests en `tests/unit/test_xgboost.py` — todos pasan

#### Tarea 3.2: Registrar XGBoost ✅

- [x] `register_model("xgboost", XGBoostSpec)` con try/except para import opcional
- [x] Config YAML sección `models.xgboost` (ya estaba presente)

**Criterio de done fase 3**: ✅ 265 passed, 2 skipped, 0 failed. `from boa_forecaster import XGBoostSpec` funciona. `get_model_spec("xgboost")` registrado.

---

### FASE 4 — LightGBM ✅ COMPLETADO (2026-03-26)

**Objetivo**: LightGBM con num_leaves como parámetro principal (diferencia clave vs XGBoost).

**Resultado**: 325 tests pasando (acumulado), 0 fallos. `from boa_forecaster import LightGBMSpec` funciona. `get_model_spec("lightgbm")` registrado.

#### Tarea 4.1: `models/lightgbm.py` — LightGBMSpec ✅

- [x] Import guard para `lightgbm`
- [x] `name = "lightgbm"`, `needs_features = True`
- [x] `__init__(self, feature_config, forecast_horizon=12, early_stopping_rounds=20)`
- [x] `search_space` — LightGBM usa `num_leaves` como parámetro principal, no `max_depth`
- [x] Constraint en `suggest_params`: si `max_depth > 0`, clipa `num_leaves <= 2^max_depth - 1`
- [x] `warm_starts`: `reg_alpha: 1e-8` (log-scale floor, no `0.0`)
- [x] `evaluate`: walk-forward 3 folds con `lgb.LGBMRegressor(verbose=-1)` + early stopping callbacks
- [x] 60 tests en `tests/unit/test_lightgbm.py` — todos pasan

#### Tarea 4.2: Registrar LightGBM ✅

- [x] `register_model("lightgbm", LightGBMSpec)` con try/except en `models/__init__.py`
- [x] `LightGBMSpec` exportado en `__init__.py` y `__all__`
- [x] `LGBM_DEFAULT_N_ESTIMATORS`, `LGBM_DEFAULT_NUM_LEAVES` añadidos a `config.py`
- [x] Config YAML sección `models.lightgbm` ya estaba presente

---

### FASE 5 — Benchmarks & comparación multi-modelo ✅ COMPLETADO (2026-03-26)

**Objetivo**: `run_benchmark_comparison` acepta múltiples modelos optimizados y produce tabla comparativa.

**Resultado**: 365 tests pasando (acumulado), 0 fallos. `from boa_forecaster import run_model_comparison, summary_table` funciona.

#### Tarea 5.1: Rediseñar `benchmarks.py` ✅

- [x] Mantener las 3 funciones de baseline existentes (`seasonal_naive`, `ets_model`, `auto_arima_nixtla`) sin cambios
- [x] Crear `run_model_comparison(df, group_cols, target_col, date_col, model_specs, n_calls_per_model, n_folds, test_size, min_train_size, m, freq) -> pd.DataFrame`:
  - Acepta `model_specs: list[ModelSpec]`
  - Para cada grupo y cada modelo: `optimize_model()` → `build_forecaster()` → `walk_forward_validation()`
  - Incluye baselines automáticamente (seasonal_naive, ETS, autoARIMA)
  - Devuelve DataFrame con columna adicional `"optimized"` (bool) para distinguir baselines de modelos optimizados
- [x] Mantener `run_benchmark_comparison()` existente como wrapper deprecado para compatibilidad v1
- [x] Actualizar `summary_table()` para funcionar con el nuevo DataFrame ampliado
- [x] Tests en `tests/unit/test_benchmarks_v2.py`:
  - Test `run_model_comparison` con `[SARIMASpec(...), RandomForestSpec()]` en synthetic data
  - Test `summary_table` con modelos mixtos (optimizados + baselines)
  - Test que `run_benchmark_comparison` v1 sigue funcionando

#### Tarea 5.2: Integration test multi-modelo ✅

- [x] Crear `tests/integration/test_multi_model.py`:
  - Test end-to-end: carga datos sintéticos → optimiza SARIMA + RF → compara → RF no explota
  - Test que `summary_table` produce `beats_naive` correcto para todos los modelos
  - Test de reproducibilidad: mismo `seed` produce mismos `best_params`

---

### FASE 6 — Package rename + API pública

**Objetivo**: `boa_forecaster` es el paquete principal; `sarima_bayes` es deprecated.

#### Tarea 6.1: Paquete `boa_forecaster` completo

- [x] Verificar que todos los imports internos son `from boa_forecaster.X import Y`
- [x] `__init__.py` exporta todos los símbolos del §3.4
- [x] Ejecutar `python -c "from boa_forecaster import *"` sin errores
- [x] Ejecutar `python -c "from boa_forecaster import optimize_model, MODEL_REGISTRY, RandomForestSpec"` sin errores

#### Tarea 6.2: Alias deprecado `sarima_bayes`

- [x] `src/sarima_bayes/__init__.py` solo contiene `DeprecationWarning` + reexportación
- [x] Test: `import sarima_bayes` levanta `DeprecationWarning`
- [x] Test: `from sarima_bayes import optimize_arima` sigue funcionando

#### Tarea 6.3: Actualizar `pyproject.toml`

- [x] Entry point principal: `boa_forecaster`
- [x] `sarima_bayes` como alias en `[tool.setuptools.packages.find]`
- [x] Bump version a `2.0.0`
- [x] Actualizar descripción del paquete: "Bayesian TPE optimization framework for time series forecasting"

---

### FASE 7 — Documentación & CI

#### Tarea 7.1: Actualizar `CLAUDE.md` ✅

- [x] Actualizar sección Architecture con nuevo árbol de módulos
- [x] Añadir sección "Adding a new model" (ver §6 de este documento)
- [x] Actualizar API notes para `optimize_model` y `ModelSpec`
- [x] Actualizar sección Commands con nuevos extras de pip

#### Tarea 7.2: Actualizar `config.example.yaml` ✅

- [x] Añadir sección `models:` completa (ver §7 de este documento)
- [x] Mantener sección `model.sarima:` existente

#### Tarea 7.3: CI matrix ✅

- [x] Mantener Python 3.9, 3.10, 3.11
- [x] Añadir job `test-ml-extras`: `pip install -e ".[ml]"` → `pytest tests/ -v`
- [x] Añadir job `test-core-only`: sin extras ML → tests que usen XGBoost/LGBM se skipean correctamente
- [x] Coverage threshold: ≥ 80% en `boa_forecaster/`

---

## 6. Diseño detallado por módulo

### 6.1 Guía para añadir un nuevo modelo

> Este proceso debe poder realizarse en menos de 3 horas para cualquier modelo sklearn-compatible.

**Paso 1**: Crear `src/boa_forecaster/models/mi_modelo.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import pandas as pd

try:
    from mi_libreria import MiModelo
    HAS_MI_MODELO = True
except ImportError:
    HAS_MI_MODELO = False

from boa_forecaster.models.base import ModelSpec, IntParam, FloatParam, CategoricalParam
from boa_forecaster.features import FeatureEngineer, FeatureConfig
from boa_forecaster.config import OPTIMIZER_PENALTY

class MiModeloSpec:
    name = "mi_modelo"
    needs_features = True  # True para ML, False para modelos AR

    def __init__(self, feature_config: FeatureConfig | None = None, forecast_horizon: int = 12):
        if not HAS_MI_MODELO:
            raise ImportError("mi_libreria no instalada. Ejecuta: pip install mi_libreria")
        self.feature_config = feature_config or FeatureConfig()
        self.forecast_horizon = forecast_horizon

    @property
    def search_space(self):
        return {
            "param1": IntParam(low=1, high=100),
            "param2": FloatParam(low=0.01, high=1.0, log=True),
            "param3": CategoricalParam(choices=["opcion_a", "opcion_b"]),
        }

    @property
    def warm_starts(self):
        return [
            {"param1": 10, "param2": 0.1, "param3": "opcion_a"},
        ]

    def suggest_params(self, trial) -> dict:
        from boa_forecaster.models.base import suggest_from_space
        return suggest_from_space(trial, self.search_space)

    def evaluate(self, series, params, metric_fn, feature_config=None) -> float:
        config = feature_config or self.feature_config
        try:
            # Implementar evaluación con walk-forward interno
            ...
        except Exception:
            return OPTIMIZER_PENALTY

    def build_forecaster(self, params, feature_config=None):
        config = feature_config or self.feature_config
        def forecaster(train: pd.Series) -> pd.Series:
            ...
        return forecaster
```

**Paso 2**: Registrar en `models/__init__.py`:
```python
try:
    from .mi_modelo import MiModeloSpec
    register_model("mi_modelo", MiModeloSpec)
except ImportError:
    pass  # dependencia opcional no instalada
```

**Paso 3**: Añadir sección en `config.example.yaml` bajo `models:`

**Paso 4**: Añadir tests en `tests/unit/test_mi_modelo.py` con:
- Test `isinstance(spec, ModelSpec)` → True
- Test `build_forecaster` devuelve Serie correcta
- Test `evaluate` no lanza excepción con datos sintéticos

### 6.2 Recursive vs Direct forecasting para ML

Se implementa **recursive** en v2.0 por simplicidad. El algoritmo:

```
Dado train de longitud T y horizonte H:
1. FeatureEngineer.fit_transform(train) → X_train, y_train
2. model.fit(X_train, y_train)
3. extended = train.copy()
4. predictions = []
5. Para h en 1..H:
   a. X_h = FeatureEngineer.transform(extended).iloc[[-1]]
   b. y_h = model.predict(X_h)[0]
   c. predictions.append(y_h)
   d. Añadir y_h a extended con fecha = last_date + h*freq
6. Retornar pd.Series(predictions, index=future_dates)
```

Limitaciones a documentar:
- Error acumulativo en horizontal largo (>12 pasos para datos mensuales)
- Sensible a la calidad de los lag features del último punto observado

**Direct forecasting** se considera mejora futura (v2.1): entrenar H modelos separados, uno por paso de horizonte.

### 6.3 Integridad temporal en CV

El único lugar donde puede ocurrir data leakage es en `FeatureEngineer`:

```
CORRECTO (un FeatureEngineer por fold):
┌─────────────────────┐  ┌─────────────────────┐
│ Fold 1              │  │ Fold 2              │
│ fe = FeatureEngineer│  │ fe = FeatureEngineer│  ← instancia nueva
│ X,y = fe.fit_transform(train_fold_1)        │  ← fit SOLO con train
│ forecasts = recursive_forecast(model, fe, .) │
└─────────────────────┘  └─────────────────────┘

INCORRECTO (mismo FeatureEngineer global):
fe = FeatureEngineer()  ← fit con todos los datos → LEAKAGE
for fold in folds:
    X,y = fe.transform(train_fold)  ← stats globales
```

El `evaluate()` de cada ModelSpec es responsable de instanciar un `FeatureEngineer` nuevo por fold.

---

## 7. Esquema de configuración YAML v2.0

```yaml
# config.example.yaml — v2.0
# Secciones existentes (sin cambios) ...

# ── NUEVA sección: modelos y sus search spaces ────────────────────────────────

models:

  # Modelo activo para la pipeline principal
  # Opciones: "sarima", "random_forest", "xgboost", "lightgbm"
  active: sarima

  sarima:
    enabled: true
    seasonal_period: 12
    search_space:
      p: {low: 0, high: 3}
      d: {low: 0, high: 2}
      q: {low: 0, high: 3}
      P: {low: 0, high: 2}
      D: {low: 0, high: 1}
      Q: {low: 0, high: 2}
    constraints:
      max_p_plus_q: 4
      max_P_plus_Q: 3
    warm_starts:
      - {p: 1, d: 1, q: 1, P: 1, D: 1, Q: 1}
      - {p: 1, d: 1, q: 0, P: 0, D: 0, Q: 0}

  random_forest:
    enabled: false
    search_space:
      n_estimators:      {type: int,         low: 50,   high: 500,  log: true}
      max_depth:         {type: int,         low: 2,    high: 20}
      min_samples_split: {type: float,       low: 0.01, high: 0.3,  log: true}
      min_samples_leaf:  {type: int,         low: 1,    high: 20}
      max_features:      {type: categorical, choices: ["sqrt", "log2", 0.5, 0.8, 1.0]}
    warm_starts:
      - {n_estimators: 100, max_depth: 5, min_samples_split: 0.1, min_samples_leaf: 1, max_features: sqrt}
      - {n_estimators: 200, max_depth: 10, min_samples_split: 0.05, min_samples_leaf: 3, max_features: log2}

  xgboost:
    enabled: false
    early_stopping_rounds: 20
    search_space:
      n_estimators:     {type: int,   low: 50,   high: 1000, log: true}
      max_depth:        {type: int,   low: 2,    high: 10}
      learning_rate:    {type: float, low: 0.005, high: 0.3, log: true}
      subsample:        {type: float, low: 0.5,  high: 1.0}
      colsample_bytree: {type: float, low: 0.5,  high: 1.0}
      min_child_weight: {type: int,   low: 1,    high: 20}
      reg_alpha:        {type: float, low: 1e-8, high: 10.0, log: true}
      reg_lambda:       {type: float, low: 1e-8, high: 10.0, log: true}
      gamma:            {type: float, low: 0.0,  high: 5.0}
    warm_starts:
      - {n_estimators: 100, max_depth: 6, learning_rate: 0.1, subsample: 0.8,
         colsample_bytree: 0.8, min_child_weight: 1, reg_alpha: 0.0, reg_lambda: 1.0, gamma: 0.0}

  lightgbm:
    enabled: false
    early_stopping_rounds: 20
    search_space:
      n_estimators:      {type: int,   low: 50,   high: 1000, log: true}
      num_leaves:        {type: int,   low: 8,    high: 256,  log: true}
      max_depth:         {type: int,   low: -1,   high: 15}
      learning_rate:     {type: float, low: 0.005, high: 0.3, log: true}
      subsample:         {type: float, low: 0.5,  high: 1.0}
      colsample_bytree:  {type: float, low: 0.5,  high: 1.0}
      min_child_samples: {type: int,   low: 5,    high: 100}
      reg_alpha:         {type: float, low: 1e-8, high: 10.0, log: true}
      reg_lambda:        {type: float, low: 1e-8, high: 10.0, log: true}
    warm_starts:
      - {n_estimators: 100, num_leaves: 31, max_depth: -1, learning_rate: 0.05,
         subsample: 0.8, colsample_bytree: 0.8, min_child_samples: 20,
         reg_alpha: 0.0, reg_lambda: 1.0}

# ── NUEVA sección: feature engineering para modelos ML ───────────────────────

features:
  lag_periods: [1, 2, 3, 6, 12]
  rolling_windows: [3, 6, 12]
  include_calendar: true
  include_trend: true
  include_expanding: false
```

---

## 8. Estrategia de testing

### Pirámide de tests v2.0

```
                    ┌─────────────────┐
                    │  Integration     │  tests/integration/
                    │  (3-5 tests)     │  Multi-modelo end-to-end
                    └────────┬────────┘
               ┌─────────────▼────────────────┐
               │      Unit tests              │  tests/unit/
               │   (≥ 80% coverage each)      │
               │  test_base.py                │
               │  test_features.py            │
               │  test_sarima_spec.py         │
               │  test_optimizer_generic.py   │
               │  test_random_forest.py       │
               │  test_xgboost.py             │
               │  test_lightgbm.py            │
               │  test_benchmarks_v2.py       │
               │  [tests existentes, sin mod] │
               └──────────────────────────────┘
```

### Fixtures nuevas en `conftest.py`

```python
# tests/conftest.py — añadir a las existentes

@pytest.fixture
def ml_series():
    """Serie de 60 obs con estacionalidad para tests de ML (necesita >max_lag)."""
    np.random.seed(42)
    idx = pd.date_range("2019-01", periods=60, freq="MS")
    trend = np.linspace(100, 200, 60)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = np.random.normal(0, 5, 60)
    return pd.Series(trend + seasonal + noise, index=idx, name="CS")

@pytest.fixture
def mock_model_spec():
    """ModelSpec mínimo para testear el optimizer sin dependencias externas."""
    class SimpleSpec:
        name = "mock"
        needs_features = False

        @property
        def search_space(self):
            return {"k": IntParam(1, 5)}

        @property
        def warm_starts(self):
            return [{"k": 3}]

        def suggest_params(self, trial):
            return {"k": trial.suggest_int("k", 1, 5)}

        def evaluate(self, series, params, metric_fn, feature_config=None):
            # Mock: score = k (queremos minimizar → óptimo es k=1)
            return float(params["k"])

        def build_forecaster(self, params, feature_config=None):
            def f(train):
                return pd.Series(
                    [train.mean()] * 12,
                    index=pd.date_range(train.index[-1], periods=13, freq="MS")[1:]
                )
            return f
    return SimpleSpec()
```

### Marcadores pytest

```ini
# pyproject.toml [tool.pytest.ini_options]
markers = [
    "slow: tests que tardan >5s",
    "requires_sklearn: tests que necesitan scikit-learn",
    "requires_xgboost: tests que necesitan xgboost",
    "requires_lightgbm: tests que necesitan lightgbm",
    "integration: tests de integración multi-modelo",
]
```

---

## 9. Gestión de dependencias opcionales

### Principio: el core no falla si ML libs no están instaladas

```python
# models/__init__.py — patrón de registro seguro
from .sarima import SARIMASpec
register_model("sarima", SARIMASpec)

try:
    from .random_forest import RandomForestSpec
    register_model("random_forest", RandomForestSpec)
except ImportError:
    pass  # scikit-learn no instalado

try:
    from .xgboost import XGBoostSpec
    register_model("xgboost", XGBoostSpec)
except ImportError:
    pass

try:
    from .lightgbm import LightGBMSpec
    register_model("lightgbm", LightGBMSpec)
except ImportError:
    pass
```

### Tabla de instalación

| Uso | Comando |
|-----|---------|
| Solo SARIMA (como v1.x) | `pip install -e .` |
| SARIMA + Random Forest | `pip install -e ".[sklearn]"` |
| Todos los modelos ML | `pip install -e ".[ml]"` |
| Solo XGBoost | `pip install -e ".[xgboost]"` |
| Solo LightGBM | `pip install -e ".[lightgbm]"` |
| Dev completo | `pip install -e ".[all,dev]"` |

---

## 10. Checklist de Definition of Done por fase

### Fase 0 ✅ COMPLETADO (2026-03-26)
- [x] `src/boa_forecaster/` existe con los 6 módulos invariantes copiados
- [x] `pytest tests/ -v` → 96 passed, 2 skipped, 0 failed
- [x] `pyproject.toml` tiene extras `sklearn`, `xgboost`, `lightgbm`, `ml`, `all`
- [x] `from boa_forecaster import smape, walk_forward_validation, clip_outliers` funciona
- [x] `from sarima_bayes import optimize_arima, forecast_arima` sigue funcionando

### Fase 1 ✅ COMPLETADO (2026-03-26)
- [x] `ModelSpec` Protocol implementado con `@runtime_checkable`
- [x] `FeatureEngineer.fit_transform` no produce leakage (test explícito)
- [x] `SARIMASpec` satisface `isinstance(SARIMASpec(), ModelSpec)` → True
- [x] `optimize_model(series, SARIMASpec(...))` produce mismo resultado que `optimize_arima(series, ...)` v1.x
- [x] `optimize_arima()` emite `DeprecationWarning`
- [x] `pytest tests/ -v --cov=src/boa_forecaster` → ≥ 80% cobertura en módulos nuevos

### Fase 2 ✅ COMPLETADO (2026-03-26)
- [x] `optimize_model(ml_series, RandomForestSpec())` converge sin excepción
- [x] `build_forecaster(best_params)(train)` devuelve Serie de longitud 12 con DatetimeIndex
- [x] Test de no-leakage pasa
- [x] `pip install -e .` (sin sklearn) → `from boa_forecaster import optimize_model` no falla

### Fase 3 ✅ COMPLETADO (2026-03-26)
- [x] `optimize_model(ml_series, XGBoostSpec())` converge sin excepción
- [x] `build_forecaster(best_params)(train)` devuelve Serie de longitud 12 con DatetimeIndex
- [x] Test de no-leakage pasa
- [x] `pip install -e .` (sin xgboost) → `from boa_forecaster import optimize_model` no falla
- [x] 53 tests en `tests/unit/test_xgboost.py` — todos pasan
- [x] `from boa_forecaster import XGBoostSpec` funciona
- [x] `get_model_spec("xgboost")` registrado en MODEL_REGISTRY

### Fase 4 ✅ COMPLETADO
- [x] Mismo criterio que Fase 3, aplicado a LightGBM
- [x] Tests con `pytest.mark.skipif` cuando la lib no está instalada

### Fase 5 ✅ COMPLETADO (2026-03-26)
- [x] `run_model_comparison(..., model_specs=[SARIMASpec(), RandomForestSpec()])` produce DataFrame válido
- [x] `summary_table` muestra `beats_naive` correcto para todos los modelos
- [x] `run_benchmark_comparison()` v1 sigue funcionando (test de regresión)

### Fase 6 ✅ COMPLETADO (2026-03-26)
- [x] `from boa_forecaster import optimize_model, RandomForestSpec` funciona
- [x] `import sarima_bayes` emite `DeprecationWarning` y luego funciona
- [x] Versión en `pyproject.toml` = `2.0.0`

### Fase 7 ✅ COMPLETADO (2026-03-26)
- [x] `CLAUDE.md` refleja nueva arquitectura
- [x] `config.example.yaml` tiene sección `models:` completa
- [x] CI job `test-ml-extras` pasa
- [x] CI job `test-core-only` pasa (skip de tests ML cuando no hay extras)

---

## Apéndice A: Consideraciones futuras (v2.1+)

| Feature | Descripción | Complejidad |
|---------|-------------|-------------|
| Direct multi-step forecasting | Entrenar H modelos, uno por horizonte | Media |
| Prophet ModelSpec | Meta's Prophet con TPE | Baja (API simple) |
| N-BEATS / N-HiTS ModelSpec | Neural baselines via neuralforecast | Alta |
| Multivariado | Covariables exógenas en ML models | Media |
| Optuna storage persistence | SQLite/PostgreSQL para estudios largos | Baja |
| Optuna pruning (MedianPruner) | Eliminar trials malos temprano en ML | Media |
| Dashboard integration | `optuna-dashboard` como extra | Baja |
| AutoML mode | `optimize_all_models()` → selección automática del mejor | Alta |

---

*Documento creado: 2026-03-26*
*Versión objetivo: boa_forecaster 2.0.0*
*Autor del plan: Claude Sonnet 4.6 + Tom Cardelo*
