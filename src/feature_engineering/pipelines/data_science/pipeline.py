"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import split_data, validate_model


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=split_data,
            inputs=["X", "y", "parameters"],
            outputs=["X_train", "X_valid", "y_train", "y_valid"],
            name="split_data",
        ),
        node(
            func=validate_model,
            inputs=["X_train", "y_train", "X_valid", "y_valid", "cat_features", "params:random_state",
                    "params:model_params"],
            outputs="best_iteration",
            name="train_model_for_validation",
        ),
    ])
