"""The public fetch function."""
from ..io import reader
from mylib._util import clean


def fetch(path):
    return clean(reader.read_csv(path))
