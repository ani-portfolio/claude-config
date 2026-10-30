"""Build feature rows and write them."""
from src.feature_pipeline import store
from src.features import build_rows


def run(fs, df):
    store.features_group(fs).insert(build_rows(df))
