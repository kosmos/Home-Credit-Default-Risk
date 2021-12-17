"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from feature_engineering.pipelines import data_engineering
from feature_engineering.pipelines import data_science


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "de": data_engineering.create_pipeline(),
        "ds": data_science.create_pipeline(),
        "__default__": data_engineering.create_pipeline() + data_science.create_pipeline(),
    }
