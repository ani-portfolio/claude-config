"""Write each prediction to the predictions group."""
from src import platform


def predictions_group(fs):
    return fs.get_or_create_feature_group(*platform.PREDICTIONS)


def deployment(project):
    return project.get_model_serving().get_deployment(platform.DEPLOYMENT)
