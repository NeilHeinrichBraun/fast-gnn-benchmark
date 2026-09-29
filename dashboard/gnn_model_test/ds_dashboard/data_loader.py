import awswrangler as wr
import pandas as pd
import streamlit as st

S3_BASE_PATH = "s3://mirakl-data-science-staging/dashboards_no_retention_rule/gnn_model_test/coview-mdm/"


@st.cache_data
def load_sessions_raw() -> pd.DataFrame:
    sessions_raw_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}sessions_raw_val.parquet/", path_suffix=".parquet"
    )

    return sessions_raw_val


@st.cache_data
def load_prod_results() -> pd.DataFrame:
    prod_results_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}prod_results_val.parquet/", path_suffix=".parquet"
    )

    return prod_results_val


@st.cache_data
def load_model_results() -> pd.DataFrame:
    model_results_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}model_results_val.parquet/", path_suffix=".parquet"
    )

    return model_results_val


@st.cache_data
def load_prod_negative_only_results() -> pd.DataFrame:
    prod_negative_only_results_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}prod_negative_only_results_val.parquet/", path_suffix=".parquet"
    )

    return prod_negative_only_results_val


@st.cache_data
def load_model_negative_only_results() -> pd.DataFrame:
    model_negative_only_results_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}model_negative_only_results_val.parquet/",
        path_suffix=".parquet",
    )

    return model_negative_only_results_val


@st.cache_data
def load_product_metadata() -> pd.DataFrame:
    product_metadata_val = wr.s3.read_parquet(
        f"{S3_BASE_PATH}product_metadata_val.parquet/", path_suffix=".parquet"
    )

    return product_metadata_val
