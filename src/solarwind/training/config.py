"""Settings for the training stage.

Kept apart from `solarwind.config.Settings` on purpose: every DVC stage lists that package as a
dependency, so adding fields there would invalidate the whole data pipeline. Windows, split labels
and feature switches still come from `Settings`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainSettings(BaseSettings):
    """Training settings; override any field with a SOLARWIND_TRAIN_<NAME> environment variable."""

    model_config = SettingsConfigDict(env_prefix="SOLARWIND_TRAIN_", frozen=True, extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "solarwind-phase1"
    log_models: bool = True
    random_state: int = 42
    targets: tuple[str, ...] = ("solar_cf", "wind_cf")

    ridge_alpha: float = 1.0
    forest_trees: int = 300
    forest_min_leaf: int = 3
    boosting_learning_rate: float = 0.05
    boosting_max_iter: int = 300
    boosting_max_leaf_nodes: int = 15

    @property
    def reports_dir(self) -> Path:
        """Folder for the model comparison report (DVC outputs)."""
        return Path("reports") / "models"


@lru_cache
def get_train_settings() -> TrainSettings:
    """Return the process-wide training settings instance."""
    return TrainSettings()
