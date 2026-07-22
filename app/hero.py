"""Clean, minimal app header — restrained typography, no effects.

Rendered as plain styled markup (no iframe/WebGL). Styling lives in ui.py.
"""

import streamlit as st


def render_hero(
    title: str = "Decision-Driven Customer Retention",
    subtitle: str = "Turning churn predictions into budget-constrained decisions",
):
    st.markdown(
        '<div class="app-hero">'
        f'<div class="app-hero__title">{title}</div>'
        f'<div class="app-hero__sub">{subtitle}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
