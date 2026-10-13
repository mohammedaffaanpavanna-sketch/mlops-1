import math

import mlflow
import numpy as np
import pandas as pd
import pytest

from feature_fixtures import make_daily, make_weather
from solarwind.config import Settings
from solarwind.features.build import build_features
from solarwind.models import evaluate
from solarwind.training import run
from solarwind.training.config import TrainSettings
from solarwind.training.estimators import make_estimators
from solarwind.training.select import FEATURE_SETS

SETTINGS = Settings()
FRAME = build_features(make_daily(SETTINGS), make_weather(SETTINGS), SETTINGS)
FAST = {"forest_trees": 15, "boosting_max_iter": 15}


def fast_settings(**extra) -> TrainSettings:
    return TrainSettings(**FAST, **extra)


def test_estimators_are_the_three_expected_models_and_fit_deterministically():
    estimators = make_estimators(fast_settings())
    assert list(estimators) == ["ridge", "random_forest", "gradient_boosting"]
    for name, estimator in estimators.items():
        first, columns = run.fit_model(FRAME, "wind_cf", "weather_only", estimator, SETTINGS)
        second, _ = run.fit_model(FRAME, "wind_cf", "weather_only", estimator, SETTINGS)
        assert np.allclose(first.predict(FRAME[columns]), second.predict(FRAME[columns])), name


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_fit_uses_only_the_training_rows(feature_set):
    estimator = make_estimators(fast_settings())["random_forest"]
    original, columns = run.fit_model(FRAME, "solar_cf", feature_set, estimator, SETTINGS)
    changed = FRAME.copy()
    test_rows = changed["split"] == SETTINGS.baseline_test_label
    changed.loc[test_rows, "solar_cf"] = changed.loc[test_rows, "solar_cf"] * 7 + 3
    refit, _ = run.fit_model(changed, "solar_cf", feature_set, estimator, SETTINGS)
    train = FRAME[FRAME["split"] == SETTINGS.baseline_train_label]
    assert np.allclose(original.predict(train[columns]), refit.predict(train[columns]))


def test_missing_history_is_imputed_for_ridge_and_left_to_boosting_otherwise():
    frame = FRAME.copy()
    frame.loc[frame.index[:20], "wind_cf_lag1"] = float("nan")
    for name in ("ridge", "random_forest", "gradient_boosting"):
        estimator = make_estimators(fast_settings())[name]
        fitted, columns = run.fit_model(frame, "wind_cf", "with_history", estimator, SETTINGS)
        assert np.isfinite(fitted.predict(frame[columns])).all(), name


def test_run_all_covers_every_model_for_every_target():
    train = fast_settings()
    long, runs = run.run_all(FRAME, SETTINGS, train)
    per_target = 3 + len(FEATURE_SETS) * 3
    assert len(runs) == len(train.targets) * per_target
    assert long.groupby(["target", "baseline"]).size().eq(len(FRAME)).all()
    assert {r.kind for r in runs} == {"baseline", "ml"}
    assert all(":" in r.label for r in runs if r.kind == "ml")


def test_comparison_table_is_relative_to_persistence_and_sorted_by_error():
    long, _ = run.run_all(FRAME, SETTINGS, fast_settings(targets=("solar_cf",)))
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    assert table["mae"].is_monotonic_increasing
    persistence = table[table["model"] == run.PERSISTENCE]["mae_vs_persistence_pct"]
    assert persistence.iloc[0] == pytest.approx(0.0)
    assert set(run.best_ml(table)) == {"solar_cf"} and ":" in run.best_ml(table)["solar_cf"]
    assert table["mae"].map(math.isfinite).all()


def test_metrics_use_the_same_rows_for_baselines_and_ml_models():
    long, _ = run.run_all(FRAME, SETTINGS, fast_settings(targets=("wind_cf",)))
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    test = primary[primary["split"] == SETTINGS.baseline_test_label]
    assert test["n"].nunique() == 1


def test_log_runs_writes_one_mlflow_run_per_model_with_data_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("solar_cf",),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        log_models=False,
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    logged = mlflow.search_runs(experiment_names=[train.mlflow_experiment])
    assert len(logged) == len(runs)
    assert set(logged["tags.data_md5"]) == {"abc123"}
    assert {"baseline", "ml"} == set(logged["tags.kind"])
    assert logged["metrics.test_mae"].notna().all()
    assert logged["metrics.train_mae"].notna().all()
    ml = logged[logged["tags.kind"] == "ml"]
    assert ml["params.n_features"].astype(int).gt(0).all()


def test_every_logged_model_loads_back_and_predicts_the_same(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(
        targets=("wind_cf",), mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    long, runs = run.run_all(FRAME, SETTINGS, train)
    runs = [r for r in runs if r.kind == "ml" and r.feature_set == "with_history"]
    assert len(runs) == 3
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    run.log_runs(runs, long, primary, FRAME, SETTINGS, train, "abc123")
    logged = mlflow.search_runs(experiment_names=[train.mlflow_experiment])
    by_name = dict(zip(logged["tags.model"], logged["run_id"], strict=True))
    for r in runs:
        model = mlflow.sklearn.load_model(f"runs:/{by_name[r.model_name]}/model")
        expected = r.fitted.predict(FRAME[r.columns])
        assert np.allclose(model.predict(FRAME[r.columns]), expected), r.model_name


def test_write_reports_creates_the_dvc_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = fast_settings(targets=("solar_cf", "wind_cf"))
    long, _ = run.run_all(FRAME, SETTINGS, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = run.comparison_table(primary, SETTINGS.baseline_test_label)
    run.write_reports(long, primary, table, SETTINGS, train)
    out = tmp_path / "reports" / "models"
    for name in ("metrics.json", "comparison.csv", "predictions.csv", "README.md"):
        assert (out / name).stat().st_size > 0, name
    assert (out / "figures" / "best.png").stat().st_size > 0
    assert "mae_vs_persistence_pct" in (out / "README.md").read_text()
    assert pd.read_csv(out / "comparison.csv")["target"].nunique() == 2
