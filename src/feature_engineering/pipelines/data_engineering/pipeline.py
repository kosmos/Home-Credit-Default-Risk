"""
This is a boilerplate pipeline 'clean'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import clean_days_employed


def create_pipeline(**kwargs):
    return Pipeline({
        node(
            func=clean_days_employed,
            inputs="application_train_dataset",
            outputs="clean_train",
            name="clean_train",
        ),
        node(
            func=clean_days_employed,
            inputs="application_test_dataset",
            outputs="clean_test",
            name="clean_test",
        ),
    })
