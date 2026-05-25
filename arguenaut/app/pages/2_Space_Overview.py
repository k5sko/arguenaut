"""Space Overview — the full perspective scatter for the last per-prompt discovery.

Reads the in-session Discovery produced on the main Hypothesis Explorer page
(st.session_state["discovery"]) — no corpus/PCA cache involved.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Arguenaut — space overview", layout="wide")


def _axis_title(disc, idx: int) -> str:
    if idx < len(disc.axes):
        return f"PC{idx + 1}: {disc.axes[idx].label}"
    return f"PC{idx + 1}"


def main() -> None:
    st.title("Space overview")
    st.caption("The full perspective cloud for your last prompt, along the discovered axes.")

    disc = st.session_state.get("discovery")
    if disc is None:
        st.info("Run a discovery on the **Hypothesis Explorer** (main) page first, then come back here.")
        return

    st.markdown(f"**Prompt:** {disc.prompt}")
    st.caption(f"model `{disc.model_id}` · layer {disc.layer}/{disc.n_layers} · {len(disc.perspectives)} perspectives")

    n_pc = len(disc.explained_variance_ratio)
    df = pd.DataFrame(
        {
            **{f"PC{i + 1}": [p.scores[i] for p in disc.perspectives] for i in range(n_pc)},
            "stance": [p.stance for p in disc.perspectives],
            "text": [p.text for p in disc.perspectives],
        }
    )
    pcs = [f"PC{i + 1}" for i in range(n_pc)]
    mode = st.radio("Plot", ["2D scatter", "3D scatter"], horizontal=True)

    if mode == "2D scatter" or n_pc < 3:
        c1, c2 = st.columns(2)
        xi = c1.selectbox("X", range(n_pc), format_func=lambda i: pcs[i], index=0)
        yi = c2.selectbox("Y", range(n_pc), format_func=lambda i: pcs[i], index=min(1, n_pc - 1))
        fig = px.scatter(
            df, x=pcs[xi], y=pcs[yi], color="stance",
            hover_data={"text": True}, opacity=0.85,
            labels={pcs[xi]: _axis_title(disc, xi), pcs[yi]: _axis_title(disc, yi)},
        )
        fig.update_traces(marker={"size": 11, "line": {"width": 0.5, "color": "white"}})
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(height=650, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          legend={"orientation": "h", "yanchor": "bottom", "y": -0.25})
        st.plotly_chart(fig, use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        xi = c1.selectbox("X", range(n_pc), format_func=lambda i: pcs[i], index=0)
        yi = c2.selectbox("Y", range(n_pc), format_func=lambda i: pcs[i], index=1)
        zi = c3.selectbox("Z", range(n_pc), format_func=lambda i: pcs[i], index=2)
        fig = px.scatter_3d(
            df, x=pcs[xi], y=pcs[yi], z=pcs[zi], color="stance",
            hover_data={"text": True}, opacity=0.85,
            labels={pcs[xi]: f"PC{xi+1}", pcs[yi]: f"PC{yi+1}", pcs[zi]: f"PC{zi+1}"},
        )
        fig.update_traces(marker={"size": 5})
        fig.update_layout(height=720, margin={"l": 0, "r": 0, "t": 30, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    # Explained-variance bar.
    ev = disc.explained_variance_ratio
    bar = go.Figure(go.Bar(x=pcs, y=ev))
    bar.update_layout(title="Variance explained by each component", yaxis_title="fraction",
                      height=260, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(bar, use_container_width=True)

    with st.expander("All perspectives"):
        st.dataframe(df[["stance", "text", *pcs]], use_container_width=True, hide_index=True)


main()
