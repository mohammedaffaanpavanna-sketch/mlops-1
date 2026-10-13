"""Train the ML models on features.csv, compare them with the baselines, log everything to MLflow.

Every model is fitted on the training rows only and scored on the same common row set as the
baselines (rows where every model predicts and the actual exists), so the numbers are comparable.
Model labels are `<model>:<feature_set>` for the ML models and the baseline names otherwise.
"""

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.base import clone

from solarwind.config import Settings, get_settings
from solarwind.logging_setup import configure_logging
from solarwind.models import evaluate, report
from solarwind.models.evaluate import LONG_COLUMNS
from solarwind.models.metrics import METRIC_NAMES
from solarwind.models.run_baselines import nested
from solarwind.training import plots
from solarwind.training.config import TrainSettings, get_train_settings
from solarwind.training.estimators import make_estimators
from solarwind.training.select import FEATURE_SETS, feature_columns

PERSISTENCE = "persistence"
PIP_REQUIREMENTS = [f"{name}=={version(name)}" for name in ("scikit-learn", "pandas", "numpy")]
# MLflow saves sklearn models with skops by default (no arbitrary pickle code). skops refuses types
# it cannot vouch for, so the ones our own fitted pipelines contain are listed here. Only load
# models from this project's own tracking store: tree node storage is not bounds-checked, so a
# tampered model file could crash or misread memory on predict.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


@dataclass
class ModelRun:
    """One model for one target: what to log about it."""

    target: str
    label: str
    kind: str  # "baseline" or "ml"
    feature_set: str
    model_name: str
    fitted: object | None = None
    columns: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def fit_model(
    frame: pd.DataFrame, target: str, feature_set: str, estimator, settings: Settings
) -> tuple[object, list[str]]:
    """Fit a fresh copy of the estimator on the training rows; return it and its columns."""
    columns = feature_columns(frame, target, feature_set, settings)
    train = frame[(frame["split"] == settings.baseline_train_label) & frame[target].notna()]
    return clone(estimator).fit(train[columns], train[target]), columns


def predict_long(
    frame: pd.DataFrame, fitted, columns: list[str], target: str, label: str
) -> pd.DataFrame:
    """One row per date with the prediction and the actual value (the baselines' long layout)."""
    return pd.DataFrame(
        {
            "date": frame["date"],
            "split": frame["split"],
            "target": target,
            "baseline": label,
            "prediction": fitted.predict(frame[columns]),
            "actual": frame[target],
        }
    )[LONG_COLUMNS]


def simple_params(model) -> dict:
    """The scalar hyperparameters of a fitted estimator, as MLflow params."""
    scalar = (int, float, str, bool, type(None))
    return {k: v for k, v in model.get_params().items() if isinstance(v, scalar)}


def run_all(
    frame: pd.DataFrame, settings: Settings, train: TrainSettings
) -> tuple[pd.DataFrame, list[ModelRun]]:
    """Fit the baselines and every ML model per target; return the long predictions and runs."""
    longs: list[pd.DataFrame] = []
    runs: list[ModelRun] = []
    for target in train.targets:
        baselines = evaluate.fit_baselines(frame, target, settings)
        longs.append(evaluate.predict_long(frame, baselines, target))
        runs += [ModelRun(target, b.name, "baseline", "none", b.name) for b in baselines]
        for feature_set in FEATURE_SETS:
            for name, estimator in make_estimators(train).items():
                fitted, columns = fit_model(frame, target, feature_set, estimator, settings)
                label = f"{name}:{feature_set}"
                longs.append(predict_long(frame, fitted, columns, target, label))
                params = simple_params(fitted.named_steps["model"])
                runs.append(
                    ModelRun(target, label, "ml", feature_set, name, fitted, columns, params)
                )
    return pd.concat(longs, ignore_index=True), runs


def comparison_table(primary: pd.DataFrame, test_label: str) -> pd.DataFrame:
    """Test metrics per target and model, best first, with MAE relative to persistence."""
    test = primary[primary["split"] == test_label].copy()
    reference = test[test["baseline"] == PERSISTENCE].set_index("target")["mae"]
    test["mae_vs_persistence_pct"] = 100.0 * (test["mae"] / test["target"].map(reference) - 1.0)
    columns = ["target", "model", "n", "mae", "rmse", "r2", "mae_pct_of_mean", "bias"]
    out = test.rename(columns={"baseline": "model"})
    return out.sort_values(["target", "mae"])[[*columns, "mae_vs_persistence_pct"]].reset_index(
        drop=True
    )


def best_ml(table: pd.DataFrame) -> dict[str, str]:
    """The lowest-MAE ML model per target (labels contain a colon, baselines do not)."""
    ml = table[table["model"].str.contains(":")]
    return ml.groupby("target").first()["model"].to_dict()


def readme_text(table: pd.DataFrame, coverage: pd.DataFrame, settings: Settings) -> str:
    """The generated report: coverage, the comparison table and how to read it."""
    lines = [
        "# Model comparison (baselines vs ML)",
        "",
        "One model per target; targets are daily capacity factors. Every model is fitted on the",
        f"`{settings.baseline_train_label}` rows only and scored on the common row set (rows where",
        "every model predicts and the actual exists). Bias is mean(prediction - actual).",
        "",
        "- `weather_only`: calendar and own-source weather. Usable by a household estimator,",
        "  because it needs no recent generation.",
        "- `with_history`: adds the target's past-day lags and trailing mean. This is the",
        "  one-day-ahead setting, where persistence (yesterday's CF) is the hard baseline.",
        "- Hyperparameters are the defaults in `TrainSettings`; nothing is tuned on the test",
        "  split.",
        "",
        "## Common row set",
        "",
        report.markdown_table(coverage),
        "",
        "## Test split, best first",
        "",
        report.markdown_table(table),
        "",
        "Negative `mae_vs_persistence_pct` means lower error than persistence.",
        "The whole test window lies in the interpolated part of the capacity series, so bias in",
        "the CF targets can come from capacity as well as from the models.",
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    long: pd.DataFrame,
    primary: pd.DataFrame,
    table: pd.DataFrame,
    settings: Settings,
    train: TrainSettings,
) -> None:
    """Write metrics.json, comparison.csv, predictions.csv, figures and the README."""
    out = train.reports_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    common = evaluate.common_rows(long)
    coverage = evaluate.coverage(long, common)
    text = json.dumps(nested(primary), indent=2, sort_keys=True) + "\n"
    (out / "metrics.json").write_text(text, newline="\n")
    table.to_csv(out / "comparison.csv", index=False, lineterminator="\n")
    long.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    plots.plot_best(
        long, best_ml(table), settings.baseline_test_label, out / "figures" / "best.png"
    )
    (out / "README.md").write_text(readme_text(table, coverage, settings), newline="\n")


def data_md5(path: Path) -> str:
    """MD5 of the feature table, the data version recorded on every MLflow run."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 (a version label, not security)


def git_commit() -> str:
    """Current Git commit, or `unknown` outside a repository."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def log_runs(
    runs: list[ModelRun],
    long: pd.DataFrame,
    primary: pd.DataFrame,
    frame: pd.DataFrame,
    settings: Settings,
    train: TrainSettings,
    features_md5: str,
) -> None:
    """One MLflow run per (target, model): params, metrics, plots and the fitted model."""
    mlflow.set_tracking_uri(train.mlflow_tracking_uri)
    mlflow.set_experiment(train.mlflow_experiment)
    commit = git_commit()
    for run in runs:
        rows = long[(long["target"] == run.target) & (long["baseline"] == run.label)]
        with (
            mlflow.start_run(run_name=f"{run.target}/{run.label}"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            mlflow.set_tags(
                {
                    "target": run.target,
                    "model": run.model_name,
                    "kind": run.kind,
                    "feature_set": run.feature_set,
                    "data_md5": features_md5,
                    "git_commit": commit,
                }
            )
            mlflow.log_params(
                {
                    "train_start": str(settings.train_start),
                    "train_end": str(settings.train_end),
                    "test_start": str(settings.test_start),
                    "test_end": str(settings.test_end),
                    "n_features": len(run.columns),
                    **{f"hp_{k}": v for k, v in run.params.items()},
                }
            )
            scores = primary[(primary["target"] == run.target) & (primary["baseline"] == run.label)]
            for row in scores.to_dict("records"):
                mlflow.log_metrics(
                    {f"{row['split']}_{m}": float(row[m]) for m in METRIC_NAMES if pd.notna(row[m])}
                )
            tmp_dir = Path(tmp)
            rows.to_csv(tmp_dir / "predictions.csv", index=False)
            plots.plot_run(rows, settings.baseline_test_label, tmp_dir / "test_period.png")
            mlflow.log_artifact(str(tmp_dir / "predictions.csv"))
            mlflow.log_artifact(str(tmp_dir / "test_period.png"))
            if run.columns:
                mlflow.log_text("\n".join(run.columns) + "\n", "features.txt")
            if run.fitted is not None and train.log_models:
                example = frame.loc[frame["split"] == settings.baseline_train_label, run.columns]
                mlflow.sklearn.log_model(
                    run.fitted,
                    name="model",
                    input_example=example.head(3),
                    pip_requirements=PIP_REQUIREMENTS,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                )


def main() -> None:
    """Command line entry point used by the train DVC stage."""
    configure_logging()
    settings, train = get_settings(), get_train_settings()
    frame = pd.read_csv(settings.features_path, parse_dates=["date"])
    long, runs = run_all(frame, settings, train)
    primary = evaluate.metrics_table(evaluate.common_rows(long))
    table = comparison_table(primary, settings.baseline_test_label)
    write_reports(long, primary, table, settings, train)
    log_runs(runs, long, primary, frame, settings, train, data_md5(settings.features_path))
    logger.info("training done", runs=len(runs), reports=str(train.reports_dir))


if __name__ == "__main__":
    main()
