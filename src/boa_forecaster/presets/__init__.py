"""Domain-specific preset packs. Explicit import required — not re-exported at top level."""

from boa_forecaster.presets.air_quality import (
    ICA_EDGES_PM25_CO2017,
    ICA_EDGES_PM25_USAQI,
    ICA_LABELS_6,
    ICA_WEIGHTS_HEALTH,
    hit_rate_ica,
    hit_rate_ica_weighted,
)

__all__ = [
    "hit_rate_ica",
    "hit_rate_ica_weighted",
    "ICA_EDGES_PM25_CO2017",
    "ICA_EDGES_PM25_USAQI",
    "ICA_LABELS_6",
    "ICA_WEIGHTS_HEALTH",
]
