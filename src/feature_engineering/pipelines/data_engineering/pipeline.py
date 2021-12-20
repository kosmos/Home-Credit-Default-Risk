"""
This is a boilerplate pipeline 'clean'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import clean_days_employed, drop_target, transform_categorical_features, fe_poly_features, \
    fe_domain_features, convert_types, agg_numeric, merge_with_main_datasets, agg_categorical, aggregate_client, \
    remove_missing_columns


def create_pipeline(**kwargs):
    return Pipeline({
        # Base clean data
        # node(
        #     func=convert_types,
        #     inputs="application_train_dataset",
        #     outputs="application_test_converted_types",
        #     name="convert_types_application_train",
        # ),
        # node(
        #     func=convert_types,
        #     inputs="application_test_dataset",
        #     outputs="application_test_converted_types",
        #     name="convert_types_application_train",
        # ),
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

        # Split Feature/Target
        node(
            func=drop_target,
            inputs=["clean_train", "params:target_column"],
            outputs=["X_transformed", "y"],
            name="drop_target",
        ),

        # FE Add polinomial features
        node(
            func=fe_poly_features,
            inputs=["X_transformed", "clean_test", "params:poly_features"],
            outputs=["train_with_poly", "test_with_poly"],
            name="fe_poly_features",
        ),

        # FE Add Domain features
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

        # FE Add features from previous_application_dataset
        node(
            func=convert_types,
            inputs="previous_application_dataset",
            outputs="previous_application_converted",
            name="convert_types_previous_application",
        ),
        node(
            func=agg_numeric,
            inputs=["previous_application_converted", "params:parent_var", "params:previous_application_prefix"],
            outputs="previous_application_numeric",
            name="agg_numeric_previous_application",
        ),
        node(
            func=agg_categorical,
            inputs=["previous_application_converted", "params:parent_var", "params:previous_application_prefix"],
            outputs="previous_application_categorical",
            name="agg_categorical_previous_application",
        ),
        node(
            func=merge_with_main_datasets,
            inputs=["train_with_domain", "test_with_domain", "previous_application_numeric"],
            outputs=["train_with_previous_application_numeric", "test_with_previous_application_numeric"],
        ),
        node(
            func=merge_with_main_datasets,
            inputs=["train_with_previous_application_numeric", "test_with_previous_application_numeric", "previous_application_categorical"],
            outputs=["train_with_previous", "test_with_previous"],
        ),

        # FE Add features from credit_card_balance_dataset
        node(
            func=convert_types,
            inputs="credit_card_balance_dataset",
            outputs="credit_card_balance_dataset_converted",
            name="convert_types_credit_card_balance",
        ),
        node(
            func=aggregate_client,
            inputs=["credit_card_balance_dataset_converted", "params:group_vars", "params:df_names_for_credit_card_balance_dataset"],
            outputs="cash_by_client",
            name="aggregate_client_cash_by_client",
        ),
        node(
            func=merge_with_main_datasets,
            inputs=["train_with_previous", "test_with_previous",
                    "cash_by_client"],
            outputs=["train_with_cash_by_client", "test_with_cash_by_client"],
        ),

        # FE Add features from pos_cash_balance_dataset
        node(
            func=convert_types,
            inputs="pos_cash_balance_dataset",
            outputs="pos_cash_balance_dataset_converted",
            name="convert_types_pos_cash_balance_dataset",
        ),
        node(
            func=aggregate_client,
            inputs=["pos_cash_balance_dataset_converted", "params:group_vars", "params:df_names_for_pos_cash_balance_dataset"],
            outputs="pos_cash_balance_by_client",
            name="aggregate_pos_cash_balance",
        ),
        node(
            func=merge_with_main_datasets,
            inputs=["train_with_cash_by_client", "test_with_cash_by_client",
                    "pos_cash_balance_by_client"],
            outputs=["train_with_pos_cash_balance_by_client", "test_with_pos_cash_balance_by_client"],
        ),

        # Drop columns with big number of empty data
        node(
            func=remove_missing_columns,
            inputs=["train_with_pos_cash_balance_by_client", "test_with_pos_cash_balance_by_client", "params:drop_column_threshold"],
            outputs=["train_without_empty", "test_without_empty"],
            name="remove_missing_columns",
        ),

        # Transform categorical features for CatBoost algorithm
        node(
            func=transform_categorical_features,
            inputs="train_without_empty",
            outputs=["X", "cat_features"],
            name="transform_categorical_features_train",
        ),
        node(
            func=transform_categorical_features,
            inputs="test_without_empty",
            outputs=["X_test", "cat_features_"],
            name="transform_categorical_features_test",
        ),
    })
