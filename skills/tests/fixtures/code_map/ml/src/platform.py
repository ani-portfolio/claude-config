"""Names of what the pipelines share."""
PROJECT = "p"
FEATURES = ("features", 1)
FEATURE_VIEW = ("fv", 1)
MODEL = "m"
PREDICTIONS = ("predictions", 1)


def login():
    import hopsworks
    return hopsworks.login(project=PROJECT)
SCORES = ("scores", 1)
DEPLOYMENT = "d"
