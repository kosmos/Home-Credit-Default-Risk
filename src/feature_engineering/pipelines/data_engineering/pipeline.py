"""
This is a boilerplate pipeline 'clean'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import clean_days_employed, drop_target, transform_categorical_features, fe_poly_features, \
    fe_domain_features, convert_types, agg_numeric, merge_with_main_datasets


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
        node(
            func=drop_target,
            inputs=["clean_train", "params:target_column"],
            outputs=["X_transformed", "y"],
            name="drop_target",
        ),
        node(
            func=fe_poly_features,
            inputs=["X_transformed", "clean_test", "params:poly_features"],
            outputs=["train_with_poly", "test_with_poly"],
            name="fe_poly_features",
        ),
        node(
            func=fe_domain_features,
            inputs="train_with_poly",
            outputs="train_with_domain",
            name="fe_domain_features_train",
        ),
        node(
            func=fe_domain_features,
            inputs="test_with_poly",
            outputs="test_with_domain",
            name="fe_domain_features_test",
        ),
        node(
            func=convert_types,
            inputs="previous_application_dataset",
            outputs="previous_application_converted",
            name="convert_types_previous_application",
        ),
        node(
            func=agg_numeric,
            inputs=["previous_application_converted", "params:parent_var", "params:previous_application_prefix"],
            outputs="previous_application_agg",
            name="agg_numeric_previous_application",
        ),
        node(
            func=merge_with_main_datasets,
            inputs=["train_with_domain", "test_with_domain", "previous_application_agg"],
            outputs=["train_with_previous", "test_with_previous"],
        ),
        node(
            func=transform_categorical_features,
            inputs="train_with_previous",
            outputs=["X", "cat_features"],
            name="transform_categorical_features_train",
        ),
        node(
            func=transform_categorical_features,
            inputs="test_with_previous",
            outputs=["X_test", "cat_features_"],
            name="transform_categorical_features_test",
        ),
    })
