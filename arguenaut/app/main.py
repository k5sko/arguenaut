"""Streamlit main page — Hypothesis Explorer.

Run with:
    streamlit run arguenaut/app/main.py
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from arguenaut.app.data_layer import AppData
from arguenaut.app.lambda_client import LambdaClient, LambdaUnreachable
from arguenaut.app.live_cache import LiveCache
from arguenaut.config import settings
from arguenaut.utils import normalise_hypothesis

st.set_page_config(page_title="Arguenaut — disagreement axes", layout="wide")


@st.cache_resource
def get_data() -> AppData:
    return AppData()


@st.cache_resource
def get_lambda_client() -> LambdaClient:
    return LambdaClient()


@st.cache_resource
def get_live_cache() -> LiveCache:
    return LiveCache()


def render_scatter(df, x: str, y: str, color_by: str = "topic", highlight=None) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = px.scatter(
        df, x=x, y=y, color=color_by,
        hover_data={"hypothesis": True, "stance": True, "text": True, x: ":.2f", y: ":.2f"},
        opacity=0.75,
    )
    fig.update_traces(marker={"size": 8, "line": {"width": 0.5, "color": "white"}})
    if highlight is not None and len(highlight) > 0:
        fig.add_trace(
            go.Scatter(
                x=[h[x] for h in highlight],
                y=[h[y] for h in highlight],
                mode="markers+text",
                marker={"size": 14, "color": "black", "symbol": "star", "line": {"width": 2, "color": "white"}},
                text=[h.get("stance", "") for h in highlight],
                textposition="top center",
                name="your hypothesis",
            )
        )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10}, height=560)
    return fig


def main() -> None:
    data = get_data()

    st.title("Arguenaut")
    st.caption(
        "An interpretability tool that discovers the fundamental axes of disagreement in scientific "
        "hypotheses by analysing the residual stream of a base LM."
    )

    available_layers = data.available_layers()
    if not available_layers:
        st.warning(
            "No PCA cache found. Run the pipeline first:\n\n"
            "```\npython -m arguenaut.scripts.extract --hypotheses data/hypotheses.json\n"
            "python -m arguenaut.scripts.run_pca\npython -m arguenaut.scripts.label_axes\n```"
        )
        return

    # ── sidebar controls ───────────────────────────────────────────────────
    st.sidebar.header("View settings")
    layer = st.sidebar.selectbox("Layer", available_layers, index=len(available_layers) - 1)
    pca = data.pca_for_layer(layer)
    if pca is None:
        st.error(f"Layer {layer} not in PCA cache")
        return
    pc_options = [f"PC{i + 1}" for i in range(pca.n_components)]
    pc_x = st.sidebar.selectbox("X axis", pc_options, index=0)
    pc_y = st.sidebar.selectbox("Y axis", pc_options, index=min(1, len(pc_options) - 1))
    color_by = st.sidebar.radio("Colour by", ["topic", "stance"], horizontal=True)

    df = data.scores_dataframe(layer)

    # ── live query ─────────────────────────────────────────────────────────
    st.subheader("Probe a new hypothesis")
    col_input, col_btn = st.columns([5, 1])
    user_text = col_input.text_input(
        "Hypothesis (a single scientific claim)",
        placeholder="e.g. Hippocampal indexing theory explains episodic recall better than reinstatement.",
    )
    go_btn = col_btn.button("Probe", type="primary")

    highlight_rows: list[dict] = []
    if go_btn and user_text:
        try:
            cleaned = normalise_hypothesis(user_text)
        except ValueError as e:
            st.error(str(e))
            return

        cache = get_live_cache()
        analysis = cache.get(cleaned)
        if analysis is not None:
            st.info("Loaded from on-disk cache (no Lambda call).")
        else:
            client = get_lambda_client()
            with st.spinner(f"Hitting Lambda backend at {settings.lambda_api_url} …"):
                try:
                    analysis = client.analyze(cleaned)
                    cache.put(analysis)
                except LambdaUnreachable as e:
                    st.error(
                        f"Could not reach Lambda backend at `{settings.lambda_api_url}` ({e}). "
                        "Start it with `python -m arguenaut.app.lambda_server` on your GPU box, "
                        "or update LAMBDA_API_URL in your `.env`."
                    )
                    analysis = None

        if analysis is not None:
            st.success(f"Got {len(analysis.perspectives)} perspectives from {analysis.model_id}")
            persp_table = []
            score_rows: list[dict] = []
            for p in analysis.perspectives:
                proj = pca.project(p.last_token[layer])[0]
                row = {"stance": p.stance, "text": p.text}
                for i in range(pca.n_components):
                    row[f"PC{i + 1}"] = float(proj[i])
                persp_table.append(row)
                score_rows.append({**row, "hypothesis": user_text, "topic": "your input"})
            st.dataframe(persp_table, use_container_width=True)
            highlight_rows = score_rows

            # Per-axis bar chart of mean activation across perspectives
            mean_scores = np.mean(
                [[r[f"PC{i + 1}"] for i in range(pca.n_components)] for r in persp_table],
                axis=0,
            )
            axes_meta = {(a.layer, a.component_idx): a for a in data.axes(layer=layer)}
            bar_labels = []
            for i in range(pca.n_components):
                a = axes_meta.get((layer, i))
                label = a.label if a and a.label else f"PC{i + 1} (unlabelled)"
                bar_labels.append(label)
            bar_fig = go.Figure(go.Bar(x=bar_labels, y=mean_scores))
            bar_fig.update_layout(
                title="Mean projection of your hypothesis onto each axis",
                yaxis_title="projection",
                margin={"l": 10, "r": 10, "t": 40, "b": 80},
                height=320,
            )
            bar_fig.update_xaxes(tickangle=-30)
            st.plotly_chart(bar_fig, use_container_width=True)

    # ── scatter of all existing perspectives ───────────────────────────────
    st.subheader(f"Disagreement space — layer {layer}, {pc_x} vs {pc_y}")
    axes_for_layer = {(a.layer, a.component_idx): a for a in data.axes(layer=layer)}
    pcx_idx = int(pc_x[2:]) - 1
    pcy_idx = int(pc_y[2:]) - 1
    ax_x = axes_for_layer.get((layer, pcx_idx))
    ax_y = axes_for_layer.get((layer, pcy_idx))
    if ax_x and ax_x.label:
        st.caption(f"**{pc_x}** ≈ {ax_x.label}  (confidence {ax_x.confidence:.2f})")
    if ax_y and ax_y.label:
        st.caption(f"**{pc_y}** ≈ {ax_y.label}  (confidence {ax_y.confidence:.2f})")
    fig = render_scatter(df, x=pc_x, y=pc_y, color_by=color_by, highlight=highlight_rows)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dataset details"):
        st.write(f"{len(df)} perspectives across {df['hypothesis'].nunique()} hypotheses")
        st.dataframe(df[["topic", "hypothesis", "stance", "text"]].head(50), use_container_width=True)


if __name__ == "__main__":
    main()
