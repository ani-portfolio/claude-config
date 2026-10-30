"""The train job."""
from src.training_pipeline import registry
from src.features import build_rows

if __name__ == "__main__":
    registry.feature_view(None)
