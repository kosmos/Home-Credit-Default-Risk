"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import split_data, transform_categorical_features, validate_model, drop_target


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=drop_target,
            inputs="clean_train",
            outputs=["X", "y"]
        ),
        node(
            func=transform_categorical_features,
            inputs="X",
            outputs=["X_transformed", "cat_features"],
            name="transform_categorical_features",
        ),
        node(
            func=split_data,
            inputs=["X_transformed", "y", "parameters"],
            outputs=["X_train", "X_valid", "y_train", "y_valid"],
            name="split_data",
        ),
        node(
            func=validate_model,
            inputs=["X_train", "y_train", "X_valid", "y_valid", "cat_features", "params:random_state", "params:model_params"],
            outputs="model",
            name="train_model_for_validation",
        ),
    ])
