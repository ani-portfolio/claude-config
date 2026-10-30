"""Load the model and answer requests."""
from src import platform, registry
from src.features import build_rows


class Predict:
    def __init__(self, project):
        self.model = registry.champion(project.get_model_registry())
        self.fv = project.get_feature_store().get_feature_view(*platform.FEATURE_VIEW)

    def lag(self, row):
        return row.get(LAG_COLUMN)


LAG_COLUMN = "temperature_lag_1d"
