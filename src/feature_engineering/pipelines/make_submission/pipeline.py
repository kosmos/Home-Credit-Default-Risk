"""
This is a boilerplate pipeline 'make_submission'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node

from .nodes import train_model, make_submission


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_model,
            inputs=["X", "y", "best_iteration", "cat_features", "params:random_state",
                    "params:model_params"],
            outputs="model",
            name="train_final_model",
        ),
        node(
            func=make_submission,
            inputs=["model", "X_test"],
            outputs="submission",
            name="make_submission",
        ),
    ])
