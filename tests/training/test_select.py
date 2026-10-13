import pytest

from feature_fixtures import make_daily, make_weather
from solarwind.config import Settings
from solarwind.features.build import build_features
from solarwind.features.calendar import COLUMNS as CALENDAR
from solarwind.training.select import FEATURE_SETS, feature_columns, source_of

SETTINGS = Settings()
FRAME = build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)
RAW_OR_TARGET = {"solar_mu", "wind_mu", "solar_capacity_mw", "wind_capacity_mw", "carry_forward"}


def test_source_of_strips_the_cf_suffix():
    assert source_of("solar_cf") == "solar" and source_of("wind_cf") == "wind"


@pytest.mark.parametrize("target", ["solar_cf", "wind_cf"])
@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_features_come_from_the_own_source_only(target, feature_set):
    source = source_of(target)
    other = "wind" if source == "solar" else "solar"
    columns = feature_columns(FRAME, target, feature_set, SETTINGS)
    assert len(columns) == len(set(columns))
    assert not [c for c in columns if c.startswith(f"{other}_")]
    assert all(c in CALENDAR or c.startswith(f"{source}_") for c in columns)


@pytest.mark.parametrize("target", ["solar_cf", "wind_cf"])
def test_weather_only_never_uses_targets_history_or_raw_generation(target):
    columns = set(feature_columns(FRAME, target, "weather_only", SETTINGS))
    assert target not in columns
    assert not columns & RAW_OR_TARGET
    assert not [c for c in columns if c.startswith(f"{target}_")]


@pytest.mark.parametrize("target", ["solar_cf", "wind_cf"])
def test_with_history_adds_exactly_the_targets_past_day_columns(target):
    weather = set(feature_columns(FRAME, target, "weather_only", SETTINGS))
    history = set(feature_columns(FRAME, target, "with_history", SETTINGS)) - weather
    assert history == {f"{target}_lag1", f"{target}_lag7", f"{target}_roll7"}


def test_wind_model_gets_wind_shear_and_solar_model_gets_the_irradiance_interaction():
    assert "wind_shear" in feature_columns(FRAME, "wind_cf", "weather_only", SETTINGS)
    assert "solar_irradiance_x_t2m" in feature_columns(FRAME, "solar_cf", "weather_only", SETTINGS)


def test_unknown_feature_set_and_missing_columns_raise():
    with pytest.raises(ValueError, match="unknown feature set"):
        feature_columns(FRAME, "solar_cf", "everything", SETTINGS)
    with pytest.raises(ValueError, match="missing columns"):
        feature_columns(FRAME.drop(columns=["solar_T2M"]), "solar_cf", "weather_only", SETTINGS)
