"""Register the best ML model for each target and assign the `best` alias.

This module is an add-on for the Phase 1 training stage. It selects the lowest-MAE
advanced model for each target from reports/models/comparison.csv, finds its MLflow
run, registers the logged model, and points alias `best` to that model version.
Run from the repository root with: python -m solarwind.training.register
"""

from __future__ import annotations

import pandas as pd
import mlflow
from mlflow import MlflowClient
from loguru import logger

from solarwind.training.config import get_train_settings


def main() -> None:
    """Register the best advanced model for solar_cf and wind_cf."""
    settings = get_train_settings()
    comparison_path = settings.reports_dir / "comparison.csv"

    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {comparison_path}. Run `python -m solarwind.training.run` first."
        )

    comparison = pd.read_csv(comparison_path)
    required_columns = {"target", "model", "mae"}
    missing = required_columns - set(comparison.columns)
    if missing:
        raise ValueError(
            f"{comparison_path} is missing required columns: {', '.join(sorted(missing))}"
        )

    # Advanced models have labels such as `ridge:with_history`; baseline labels do not
    # contain a colon. Choose the lowest test MAE among advanced models for each target.
    comparison["model"] = comparison["model"].astype(str)
    advanced = comparison[comparison["model"].str.contains(":", regex=False)].copy()
    if advanced.empty:
        raise ValueError("No advanced-model rows were found in comparison.csv.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_tracking_uri,
    )

    experiment = client.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {settings.mlflow_experiment!r} was not found. "
            "Run the training module first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.kind = 'ml'",
        max_results=1000,
    )
    if not runs:
        raise RuntimeError(
            f"No advanced-model MLflow runs found in {settings.mlflow_experiment!r}."
        )

    for target in settings.targets:
        target_rows = advanced[advanced["target"] == target].sort_values("mae")
        if target_rows.empty:
            raise RuntimeError(f"No advanced-model comparison row found for {target!r}.")

        best_label = str(target_rows.iloc[0]["model"])
        model_name, feature_set = best_label.rsplit(":", 1)
        matching_runs = [
            run
            for run in runs
            if run.data.tags.get("target") == target
            and run.data.tags.get("model") == model_name
            and run.data.tags.get("feature_set") == feature_set
        ]
        if not matching_runs:
            raise RuntimeError(
                f"Could not find an MLflow run for {target} / {best_label}. "
                "Check that training and registration use the same mlflow.db."
            )

        # If the same run has somehow been logged more than once, use its lowest test MAE.
        matching_runs.sort(key=lambda run: run.data.metrics.get("test_mae", float("inf")))
        source_run = matching_runs[0]
        registered_name = f"solarwind-{target}"
        model_uri = f"runs:/{source_run.info.run_id}/model"

        logger.info(
            "registering best model",
            target=target,
            model=best_label,
            test_mae=source_run.data.metrics.get("test_mae"),
            run_id=source_run.info.run_id,
        )
        version = mlflow.register_model(model_uri=model_uri, name=registered_name)
        client.set_registered_model_alias(
            name=registered_name,
            alias="best",
            version=str(version.version),
        )
        logger.success(
            "model registered",
            registered_model=registered_name,
            version=version.version,
            alias="best",
        )


if __name__ == "__main__":
    main()
