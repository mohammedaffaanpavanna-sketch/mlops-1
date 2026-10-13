"""Which feature columns each model sees.

Two feature sets per target, both using only that target's own source (solar models never see
wind weather and the other way round) and never the raw MU or capacity columns:

- `weather_only`: calendar and weather. This is what the household estimator can use, because it
  needs no recent generation.
- `with_history`: `weather_only` plus the target's own past-day lags and trailing mean (a
  one-day-ahead setting, the one persistence competes in).
"""

import pandas as pd

from solarwind.config import Settings
from solarwind.features.calendar import COLUMNS as CALENDAR
from solarwind.features.schema import (
    lag_columns,
    roll_columns,
    target_history_columns,
    weather_columns,
)
from solarwind.features.weather import base_weather_columns

FEATURE_SETS = ("weather_only", "with_history")


def source_of(target: str) -> str:
    """`solar_cf` -> `solar`."""
    return target.removesuffix("_cf")


def feature_columns(
    frame: pd.DataFrame, target: str, feature_set: str, settings: Settings
) -> list[str]:
    """Ordered feature columns for one target and feature set; raises if any is missing."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {FEATURE_SETS}")
    base = base_weather_columns(settings)
    weather = weather_columns(settings)
    if settings.features_weather_rolling:
        weather += roll_columns(base, settings.weather_rolling_windows)
    if settings.features_weather_lags:
        weather += lag_columns(base, settings.weather_lag_days)
    prefix = f"{source_of(target)}_"
    columns = [c for c in weather if c.startswith(prefix)]
    if settings.features_calendar:
        columns = [*CALENDAR, *columns]
    if feature_set == "with_history":
        columns += [c for c in target_history_columns(settings) if c.startswith(f"{target}_")]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    return columns
