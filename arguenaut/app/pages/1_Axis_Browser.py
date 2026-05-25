"""Axis Browser — detailed view of each axis from the last per-prompt discovery.

Reads the in-session Discovery produced on the main Hypothesis Explorer page
(st.session_state["discovery"]) — no corpus/PCA cache involved.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Arguenaut — axis browser", layout="wide")


def main() -> None:
    st.title("Axis browser")
    st.caption("Detailed view of each axis discovered for your last prompt.")

    disc = st.session_state.get("discovery")
    if disc is None:
        st.info("Run a discovery on the **Hypothesis Explorer** (main) page first, then come back here.")
        return

    st.markdown(f"**Prompt:** {disc.prompt}")
    st.caption(f"model `{disc.model_id}` · layer {disc.layer}/{disc.n_layers} · {len(disc.perspectives)} perspectives")

    # Overview table of every discovered axis.
    rows = [
        {
            "PC": f"PC{a.component_idx + 1}",
            "label": a.label,
            "high pole": a.high_pole,
            "low pole": a.low_pole,
            "explained var": round(a.explained_variance, 3),
        }
        for a in disc.axes
    ]
    st.dataframe(rows, use_container_width=True)

    # Inspect one axis in depth.
    pc_labels = [f"PC{a.component_idx + 1}" for a in disc.axes]
    selected = st.selectbox("Inspect axis", pc_labels)
    pc_idx = int(selected[2:]) - 1
    axis = next(a for a in disc.axes if a.component_idx == pc_idx)

    st.markdown(f"### {selected} — {axis.label}")
    if axis.rationale:
        st.caption(axis.rationale)
    c1, c2 = st.columns(2)
    c1.metric("▲ high pole", axis.high_pole)
    c2.metric("▼ low pole", axis.low_pole)
    c1.metric("explained variance", f"{axis.explained_variance:.3f}")

    # Distribution of every perspective's score along this axis.
    scores = [p.scores[pc_idx] for p in disc.perspectives]
    df = pd.DataFrame(
        {
            "score": scores,
            "stance": [p.stance for p in disc.perspectives],
            "text": [p.text for p in disc.perspectives],
            "pole": ["▲ high" if s >= 0 else "▼ low" for s in scores],
        }
    ).sort_values("score")

    hist = px.histogram(
        df, x="score", color="pole", nbins=24,
        color_discrete_map={"▲ high": "#ef553b", "▼ low": "#636efa"},
    )
    hist.update_layout(
        title="Where each perspective falls along this axis", height=300,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(hist, use_container_width=True)

    lo, hi = st.columns(2)
    lo.markdown(f"##### ▼ Lowest on *{axis.low_pole}*")
    lo.dataframe(df.head(5)[["score", "stance", "text"]], use_container_width=True, hide_index=True)
    hi.markdown(f"##### ▲ Highest on *{axis.high_pole}*")
    hi.dataframe(df.tail(5).iloc[::-1][["score", "stance", "text"]], use_container_width=True, hide_index=True)


main()
