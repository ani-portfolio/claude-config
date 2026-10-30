"""Which model serves."""
from src import platform


def champion(mr):
    return mr.get_model(platform.MODEL)
