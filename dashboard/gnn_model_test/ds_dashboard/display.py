"""Shared rendering helpers for the product rows.

Each row is a single horizontally scrollable flex container instead of st.columns:
st.columns always splits the available width between its columns, so a session with
many products ends up with unreadable thumbnails and one-word-per-line titles.
Fixed-width cards keep every product legible whatever the count, and the row scrolls.

Colours are passed as a list aligned with the products rather than derived from each
product: session_products is a chronological view log, so the same product can appear
several times and only its position tells the occurrences apart.
"""

import html

import pandas as pd
import streamlit as st

TRIGGER_BG = "#e3f2fd"
POSITIVE_BG = "#e6f4ea"
NEGATIVE_BG = "#fceceb"
NEUTRAL_BG = "transparent"

# ~10 cards visible on a wide screen, the rest reachable by scrolling.
CARD_WIDTH_PX = 140
IMAGE_HEIGHT_PX = 110
TITLE_MAX_LINES = 3

# Kept on single lines: indented HTML passed to st.markdown can be picked up as a
# markdown code block.
_CARD = (
    '<div style="flex:0 0 {width}px;background-color:{bg};padding:8px;'
    'border-radius:8px;box-sizing:border-box">'
    '<img src="{url}" title="{name}" style="width:100%;height:{height}px;'
    'object-fit:contain;border-radius:4px;display:block">'
    '<p title="{name}" style="margin:6px 0 0;text-align:center;font-size:0.8rem;'
    "line-height:1.25;display:-webkit-box;-webkit-line-clamp:{lines};"
    '-webkit-box-orient:vertical;overflow:hidden">{name}</p>'
    "{caption}</div>"
)

_CAPTION = (
    '<p title="{full}" style="margin:4px 0 0;text-align:center;font-size:0.7rem;'
    'color:#6b6b6b">{short}</p>'
)

_ROW = (
    '<div style="display:flex;gap:8px;overflow-x:auto;padding-bottom:8px">{cards}</div>'
)


def reco_color(product, positive_ids, negative_ids):
    """Background of a recommended product: known positive, known negative or neither."""
    if product["internal_id"] in positive_ids:
        return POSITIVE_BG
    if product["internal_id"] in negative_ids:
        return NEGATIVE_BG
    return NEUTRAL_BG


def first_occurrence_index(products, internal_id):
    """Position of the first view of a product, or None if it was never viewed."""
    return next(
        (
            i
            for i, product in enumerate(products)
            if product["internal_id"] == internal_id
        ),
        None,
    )


def _caption(product):
    """View time of a session product, empty for products that carry no timestamp."""
    emitted = product.get("emitted")
    if emitted is None:
        return ""
    stamp = pd.Timestamp(emitted)
    return _CAPTION.format(
        short=stamp.strftime("%H:%M:%S"),
        full=stamp.strftime("%Y-%m-%d %H:%M:%S"),
    )


def render_product_row(products, colors, show_view_time=False):
    """Render products as fixed-size cards in a scrollable row.

    colors must be aligned with products. Names are escaped: marketplace titles can
    contain quotes or angle brackets that would otherwise break the markup.
    """
    cards = "".join(
        _CARD.format(
            width=CARD_WIDTH_PX,
            height=IMAGE_HEIGHT_PX,
            lines=TITLE_MAX_LINES,
            bg=color,
            url=html.escape(str(product.get("imageUrl", "")), quote=True),
            name=html.escape(str(product.get("name", "Produit inconnu")), quote=True),
            caption=_caption(product) if show_view_time else "",
        )
        for product, color in zip(products, colors)
    )
    st.markdown(_ROW.format(cards=cards), unsafe_allow_html=True)
