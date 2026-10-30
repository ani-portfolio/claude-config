"""The feature store side of the feature pipeline."""
from src import platform


def features_group(fs):
    name, version = platform.FEATURES
    return fs.get_or_create_feature_group(name=name, version=version)


def view(fs):
    return fs.get_or_create_feature_view(*platform.FEATURE_VIEW)
