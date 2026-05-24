"""Axis Browser — list every discovered axis with labels, confidences, examples."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from arguenaut.app.data_layer import AppData

st.set_page_config(page_title="Arguenaut — axis browser", layout="wide")


@st.cache_resource
def get_data() -> AppData:
    return AppData()


def main() -> None:
    data = get_data()
    st.title("Axis browser")
    st.caption("Every (layer, principal component) the labeller has touched.")

    layers = data.available_layers()
    if not layers:
        st.warning("No PCA cache yet.")
        return

    layer = st.selectbox("Layer", layers, index=len(layers) - 1)
    axes = data.axes(layer=layer)
    if not axes:
        st.info("No axes recorded for this layer.")
        return

    df_rows = []
    for a in axes:
        df_rows.append(
            {
                "PC": f"PC{a.component_idx + 1}",
                "label": a.label or "—",
                "high_pole": a.high_pole or "—",
                "low_pole": a.low_pole or "—",
                "confidence": a.confidence if a.confidence is not None else float("nan"),
                "explained_var": a.explained_var if a.explained_var is not None else float("nan"),
            }
        )
    st.dataframe(df_rows, use_container_width=True)

    selected_pc = st.selectbox("Inspect axis", [r["PC"] for r in df_rows])
    pc_idx = int(selected_pc[2:]) - 1
    axis = next(a for a in axes if a.component_idx == pc_idx)

    st.markdown(f"### {selected_pc} — {axis.label or 'unlabelled'}")
    cols = st.columns(2)
    cols[0].metric("HIGH pole", axis.high_pole or "—")
    cols[1].metric("LOW pole", axis.low_pole or "—")
    cols2 = st.columns(2)
    cols2[0].metric("Verification confidence", f"{axis.confidence:.2f}" if axis.confidence is not None else "—")
    cols2[1].metric("Explained variance", f"{axis.explained_var:.3f}" if axis.explained_var is not None else "—")

    pca = data.pca_for_layer(layer)
    if pca is not None and pc_idx < pca.n_components:
        df = data.scores_dataframe(layer)
        df = df.sort_values(f"PC{pc_idx + 1}")
        st.markdown("##### Five LOWEST-scoring perspectives")
        st.dataframe(df.head(5)[["topic", "stance", f"PC{pc_idx + 1}", "text"]], use_container_width=True)
        st.markdown("##### Five HIGHEST-scoring perspectives")
        st.dataframe(df.tail(5).iloc[::-1][["topic", "stance", f"PC{pc_idx + 1}", "text"]], use_container_width=True)

        hist = px.histogram(df, x=f"PC{pc_idx + 1}", nbins=40)
        hist.update_layout(title="Distribution of projections along this axis", height=300, margin={"l":10,"r":10,"t":40,"b":10})
        st.plotly_chart(hist, use_container_width=True)

    st.markdown("##### Refinement history")
    rounds = data.axis_verifications(axis.id)
    if not rounds:
        st.caption("No verification rounds recorded.")
    else:
        for r in rounds:
            with st.expander(f"Round {r['round']} — candidate: {r['candidate']!r}  (score {r['score']:+.2f})"):
                detail = r.get("detail") or {}
                st.write(detail.get("rationale", ""))
                wh = (detail.get("counterexamples") or {}).get("wrong_high") or []
                wl = (detail.get("counterexamples") or {}).get("wrong_low") or []
                if wh:
                    st.markdown("**Predicted HIGH, scored LOW:**")
                    for t in wh:
                        st.text(t)
                if wl:
                    st.markdown("**Predicted LOW, scored HIGH:**")
                    for t in wl:
                        st.text(t)


main()
