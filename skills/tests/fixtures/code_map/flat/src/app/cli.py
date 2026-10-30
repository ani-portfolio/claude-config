"""Command line entry point."""
from app import core
from utils.text import slug


def main():
    core.run()
    slug("x")
