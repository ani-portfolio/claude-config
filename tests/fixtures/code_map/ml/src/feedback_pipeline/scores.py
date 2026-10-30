"""Join actuals to predictions and write live scores."""
from src import platform


def score(fs):
    predictions = fs.get_feature_group(*platform.PREDICTIONS).read()
    return fs.get_or_create_feature_group(*platform.SCORES).insert(predictions)
