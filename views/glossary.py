"""Glossary: every term used in the app, explained in plain language."""
from __future__ import annotations

from html import escape

import streamlit as st

from core import branding, glossary

st.markdown(branding.header_html("Glossary", "Every term in the app explained in plain language. "
                                              "Search, or browse by topic."), unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])
q = c1.text_input("Search a term (for example: sensitivity, Q factor, gauge factor, R²)", key="gl_q", placeholder="type here …")
cats = ["All topics"] + glossary.categories()
cat = c2.selectbox("Topic", cats, key="gl_cat")


def card(t: glossary.Term) -> str:
    f = f'<div class="t-formula">Formula: <code>{escape(t.formula)}</code></div>' if t.formula else ""
    return (f'<div class="term-card"><div class="t-name">{escape(t.name)}</div><div class="t-short">{escape(t.short)}</div>'
            f'<div class="t-long">{escape(t.long)}</div>{f}</div>')


if q.strip():
    hits = glossary.search(q)
    if cat != "All topics":
        hits = [t for t in hits if t.category == cat]
    st.caption(f"{len(hits)} matching term(s)")
    st.markdown("".join(card(t) for t in hits) or "No term matches. Try a shorter word.", unsafe_allow_html=True)
else:
    show = glossary.categories() if cat == "All topics" else [cat]
    st.caption(f"{len(glossary.TERMS)} terms in {len(glossary.categories())} topics. Tip: the small **?** next to every input in the "
               "Sensor Lab sidebar shows the same explanation.")
    for c in show:
        items = glossary.terms_in(c)
        with st.expander(f"{c}  ({len(items)})", expanded=(cat != "All topics")):
            st.markdown("".join(card(t) for t in items), unsafe_allow_html=True)
