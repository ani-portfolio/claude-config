"""Training data in, models out."""
from src import platform


def feature_view(fs):
    return fs.get_feature_view(*platform.FEATURE_VIEW)


def register(project, fv):
    meta = {"feature_view": list(platform.FEATURE_VIEW)}
    return project.get_model_registry().python.create_model(name=platform.MODEL, metrics=meta)


def lost(fs):
    return fs.get_feature_group(platform.MISSING)


def live_scores(fs):
    return fs.get_feature_group(*platform.SCORES)
