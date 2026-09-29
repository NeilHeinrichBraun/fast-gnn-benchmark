import streamlit as st
import pandas as pd

from ds_dashboard.data_construction import (
    sessions_display_negative_only as sessions_display,
    prod_reco_display_negative_only as prod_reco_display,
    model_reco_display_negative_only as model_reco_display,
)
from ds_dashboard.display import (
    NEUTRAL_BG,
    TRIGGER_BG,
    first_occurrence_index,
    reco_color,
    render_product_row,
)

st.set_page_config(layout="wide")

st.title("GNN dashboard - Negative only")


def _format_exec_code(code):
    name = sessions_display.loc[sessions_display["exec_code"] == code, "name"].iloc[0]
    return str(name) if pd.notna(name) else code


selected_exec_code = st.selectbox(
    label="Choisissez un produit",
    options=sessions_display["exec_code"],
    format_func=_format_exec_code,
)

selected_session = sessions_display.loc[
    sessions_display["exec_code"] == selected_exec_code, "session_products"
].iloc[0]
selected_trigger_internal_id = sessions_display.loc[
    sessions_display["exec_code"] == selected_exec_code, "trigger_internal_id"
].iloc[0]

st.subheader("Produits vus dans la session")
if not selected_session:
    st.info("Aucun produit vu dans cette session.")
else:
    # Only the earliest view of the trigger is highlighted, even when it was seen again
    # later in the session.
    trigger_index = first_occurrence_index(
        selected_session, selected_trigger_internal_id
    )
    render_product_row(
        selected_session,
        [
            TRIGGER_BG if position == trigger_index else NEUTRAL_BG
            for position in range(len(selected_session))
        ],
    )

selected_prod_returned = prod_reco_display.loc[
    prod_reco_display["exec_code"] == selected_exec_code, "products_returned"
].iloc[0]
selected_prod_positives = prod_reco_display.loc[
    prod_reco_display["exec_code"] == selected_exec_code, "positives"
].iloc[0]
selected_prod_negatives = prod_reco_display.loc[
    prod_reco_display["exec_code"] == selected_exec_code, "negatives"
].iloc[0]

positive_ids = {p["internal_id"] for p in selected_prod_positives}
negative_ids = {p["internal_id"] for p in selected_prod_negatives}

st.subheader("Produits renvoyés par la prod")
if not selected_prod_returned:
    st.info("Aucun produit renvoyé par la prod pour cette session.")
else:
    render_product_row(
        selected_prod_returned,
        [
            reco_color(product, positive_ids, negative_ids)
            for product in selected_prod_returned
        ],
    )

selected_model_returned = model_reco_display.loc[
    model_reco_display["exec_code"] == selected_exec_code, "products_returned"
].iloc[0]
selected_model_positives = model_reco_display.loc[
    model_reco_display["exec_code"] == selected_exec_code, "positives"
].iloc[0]
selected_model_negatives = model_reco_display.loc[
    model_reco_display["exec_code"] == selected_exec_code, "negatives"
].iloc[0]

model_positive_ids = {p["internal_id"] for p in selected_model_positives}
model_negative_ids = {p["internal_id"] for p in selected_model_negatives}

st.subheader("Produits renvoyés par le modèle")
if not selected_model_returned:
    st.info("Aucun produit renvoyé par le modèle pour cette session.")
else:
    render_product_row(
        selected_model_returned,
        [
            reco_color(product, model_positive_ids, model_negative_ids)
            for product in selected_model_returned
        ],
    )
