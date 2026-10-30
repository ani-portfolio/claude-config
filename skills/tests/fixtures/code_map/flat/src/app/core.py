"""The core logic of the app."""
from utils import text
import config


def run():
    return text.slug(config.NAME)
