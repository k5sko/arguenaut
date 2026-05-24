"""Space Overview — full scatter of all perspectives, with axis-pair toggles."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from arguenaut.app.data_layer import AppData

st.set_page_config(page_title="Arguenaut — space overview", layout="wide")


@st.cache_resource
def get_data() -> AppData:
    return AppData()


def main() -> None:
    data = get_data()
    st.title("Space overview")
    st.caption("The full disagreement landscape, colour-coded by topic.")

    layers = data.available_layers()
    if not layers:
        st.warning("No PCA cache yet.")
        return

    layer = st.selectbox("Layer", layers, index=len(layers) - 1)
    pca = data.pca_for_layer(layer)
    if pca is None:
        return
    df = data.scores_dataframe(layer)
    if df.empty:
        st.info("No perspectives stored.")
        return

    pcs = [f"PC{i + 1}" for i in range(pca.n_components)]
    mode = st.radio("Plot", ["2D scatter", "3D scatter"], horizontal=True)

    if mode == "2D scatter":
        c1, c2 = st.columns(2)
        x = c1.selectbox("X", pcs, index=0)
        y = c2.selectbox("Y", pcs, index=1)
        fig = px.scatter(
            df, x=x, y=y, color="topic",
            hover_data={"hypothesis": True, "stance": True, "text": True},
            symbol="stance",
            opacity=0.8,
        )
        fig.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "white"}})
        fig.update_layout(height=650, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X", pcs, index=0)
        y = c2.selectbox("Y", pcs, index=1)
        z = c3.selectbox("Z", pcs, index=min(2, len(pcs) - 1))
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color="topic",
            hover_data={"hypothesis": True, "stance": True},
            opacity=0.8,
        )
        fig.update_traces(marker={"size": 4})
        fig.update_layout(height=720, margin={"l": 0, "r": 0, "t": 30, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Filter by topic")
    topics = sorted(df["topic"].unique())
    chosen = st.multiselect("Topics", topics, default=topics)
    sub = df[df["topic"].isin(chosen)]
    st.dataframe(sub[["topic", "hypothesis", "stance", "text"]], use_container_width=True)


main()
