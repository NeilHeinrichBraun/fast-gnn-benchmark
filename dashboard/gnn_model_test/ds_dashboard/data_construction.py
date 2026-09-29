import numpy as np
import pandas as pd
import streamlit as st
from ds_dashboard.data_loader import (
    load_sessions_raw,
    load_prod_results,
    load_model_results,
    load_product_metadata,
    load_prod_negative_only_results,
    load_model_negative_only_results,
)

SEED = 56

sessions_raw_val = load_sessions_raw()

prod_results_val = load_prod_results()
model_results_val = load_model_results()

prod_negative_only_results_val = load_prod_negative_only_results()
model_negative_only_results_val = load_model_negative_only_results()

product_metadata_val = load_product_metadata()


@st.cache_data
def products_session_to_display(
    df_sessions: pd.DataFrame,
    df_product_metadata: pd.DataFrame,
    sample_number=20,
    random_state=SEED,
) -> pd.DataFrame:
    df_sessions = df_sessions.drop_duplicates(
        subset="trigger_internal_id", keep="first"
    )

    df_20 = df_sessions.sample(n=sample_number, random_state=random_state)

    metadata_lookup = df_product_metadata.set_index("internal_id")[
        ["name", "imageUrl"]
    ].to_dict(orient="index")

    df_20 = df_20.merge(
        df_product_metadata,
        left_on="trigger_internal_id",
        right_on="internal_id",
        how="left",
    ).drop(columns="internal_id")

    df_20["session_products"] = df_20["session_products"].apply(
        lambda products: [
            {**p, **metadata_lookup.get(p["internal_id"], {})} for p in products
        ]
    )

    return df_20


@st.cache_data
def prod_reco_to_display(
    df_20: pd.DataFrame,
    df_prod_results: pd.DataFrame,
    df_product_metadata: pd.DataFrame,
) -> pd.DataFrame:
    df_20_prod = df_20[["exec_code"]].merge(df_prod_results, on="exec_code", how="left")

    metadata_lookup = df_product_metadata.set_index("internal_id")[
        ["name", "imageUrl"]
    ].to_dict(orient="index")

    for col in ["positives", "negatives", "products_returned"]:
        df_20_prod[col] = df_20_prod[col].apply(
            lambda products: (
                [{**p, **metadata_lookup.get(p["internal_id"], {})} for p in products]
                if isinstance(products, (list, np.ndarray))
                else []
            )
        )

    return df_20_prod


@st.cache_data
def model_reco_to_display(
    df_20: pd.DataFrame,
    df_model_results: pd.DataFrame,
    df_product_metadata: pd.DataFrame,
) -> pd.DataFrame:
    df_20_model = df_20[["exec_code"]].merge(
        df_model_results, on="exec_code", how="left"
    )

    metadata_lookup = df_product_metadata.set_index("internal_id")[
        ["name", "imageUrl"]
    ].to_dict(orient="index")

    for col in ["positives", "negatives", "products_returned"]:
        df_20_model[col] = df_20_model[col].apply(
            lambda products: (
                [{**p, **metadata_lookup.get(p["internal_id"], {})} for p in products]
                if isinstance(products, (list, np.ndarray))
                else []
            )
        )

    return df_20_model


sessions_with_positive = sessions_raw_val[
    sessions_raw_val["exec_code"].isin(prod_results_val["exec_code"])
]
sessions_negative_only = sessions_raw_val[
    sessions_raw_val["exec_code"].isin(prod_negative_only_results_val["exec_code"])
]

sessions_display_with_positive = products_session_to_display(
    sessions_with_positive, product_metadata_val
)
sessions_display_negative_only = products_session_to_display(
    sessions_negative_only, product_metadata_val
)

prod_reco_display_with_positive = prod_reco_to_display(
    sessions_display_with_positive, prod_results_val, product_metadata_val
)
model_reco_display_with_positive = model_reco_to_display(
    sessions_display_with_positive, model_results_val, product_metadata_val
)

prod_reco_display_negative_only = prod_reco_to_display(
    sessions_display_negative_only, prod_negative_only_results_val, product_metadata_val
)
model_reco_display_negative_only = model_reco_to_display(
    sessions_display_negative_only,
    model_negative_only_results_val,
    product_metadata_val,
)
