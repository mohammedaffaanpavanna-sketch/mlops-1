"""The three ML estimators, as sklearn pipelines with the same fit/predict interface.

Ridge and the random forest get a median imputer (target-history features have gaps); the
histogram gradient boosting model handles missing values itself. Imputer medians and the scaler
are fitted on the training rows only, because the whole pipeline is fitted on them.
"""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solarwind.training.config import TrainSettings


def make_estimators(settings: TrainSettings) -> dict[str, Pipeline]:
    """Unfitted pipelines keyed by model name, in reporting order."""
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=settings.ridge_alpha)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=settings.forest_trees,
                        min_samples_leaf=settings.forest_min_leaf,
                        random_state=settings.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=settings.boosting_learning_rate,
                        max_iter=settings.boosting_max_iter,
                        max_leaf_nodes=settings.boosting_max_leaf_nodes,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        ),
    }
