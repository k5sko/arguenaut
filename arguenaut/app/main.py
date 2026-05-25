"""Streamlit main page — per-prompt disagreement-axis discovery.

Enter a single scientific hypothesis. Arguenaut generates a diverse set of
perspectives on it, runs them through the base LM on Lambda, and discovers —
live, for that specific debate — the axes along which the perspectives disagree.

Run with:
    streamlit run arguenaut/app/main.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from arguenaut.extraction import make_extractor
from arguenaut.generation import PerspectiveGenerator
from arguenaut.live import Discovery, discover_axes_for_prompt
from arguenaut.utils import normalise_hypothesis

st.set_page_config(page_title="Arguenaut — disagreement axes", layout="wide")


@st.cache_resource(show_spinner=False)
def get_generator() -> PerspectiveGenerator:
    return PerspectiveGenerator()


@st.cache_resource(show_spinner=False)
def get_extractor():
    """Remote extractor pointed at the live Lambda box (from .lambda-state.json)."""
    ext = make_extractor(remote=True)
    ext.load()  # raises if the GPU box is unreachable
    return ext


def _axis_caption(ax) -> str:
    ev = f"{ax.explained_variance * 100:.0f}% var"
    return f"**PC{ax.component_idx + 1}** · {ax.label}  _( {ev} )_"


def _wrap(text: str, width: int = 60) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


def render_scatter(disc: Discovery, x_idx: int, y_idx: int) -> go.Figure:
    rows = [
        {
            "PCx": p.scores[x_idx],
            "PCy": p.scores[y_idx],
            "stance": p.stance,
            "text": "<br>".join(_wrap(p.text)),
        }
        for p in disc.perspectives
    ]
    df = pd.DataFrame(rows)
    ax_x = disc.axes[x_idx] if x_idx < len(disc.axes) else None
    ax_y = disc.axes[y_idx] if y_idx < len(disc.axes) else None
    x_title = ax_x.label if ax_x else f"PC{x_idx + 1}"
    y_title = ax_y.label if ax_y else f"PC{y_idx + 1}"

    fig = px.scatter(
        df, x="PCx", y="PCy", color="stance",
        hover_data={"text": True, "PCx": ":.2f", "PCy": ":.2f", "stance": False},
        labels={"PCx": x_title, "PCy": y_title},
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 0.5, "color": "white"}})
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=600,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )
    return fig


def run_discovery(prompt: str) -> Discovery:
    extractor = get_extractor()
    generator = get_generator()
    return discover_axes_for_prompt(
        prompt,
        extractor=extractor,
        generator=generator,
        n_perspectives=st.session_state["n_perspectives"],
        n_axes=st.session_state["n_axes"],
        layer_frac=st.session_state["layer_frac"],
    )


def main() -> None:
    st.title("Arguenaut")
    st.caption(
        "Enter a scientific hypothesis. Arguenaut generates many perspectives on it, "
        "runs them through a base LM, and discovers the axes of disagreement **for that "
        "specific debate** — on the fly."
    )

    # ── sidebar controls ────────────────────────────────────────────────────
    st.sidebar.header("Discovery settings")
    st.session_state.setdefault("n_perspectives", 32)
    st.session_state.setdefault("n_axes", 4)
    st.session_state.setdefault("layer_frac", 0.7)
    st.session_state["n_perspectives"] = st.sidebar.slider(
        "Perspectives per prompt", 12, 48, st.session_state["n_perspectives"], step=4,
        help="More perspectives → cleaner axes, but more Groq + GPU work.",
    )
    st.session_state["n_axes"] = st.sidebar.slider(
        "Axes to discover", 2, 6, st.session_state["n_axes"]
    )
    st.session_state["layer_frac"] = st.sidebar.slider(
        "Layer depth (fraction)", 0.3, 0.95, st.session_state["layer_frac"], step=0.05,
        help="Which residual-stream layer to analyse. Mid-late layers separate best.",
    )

    # ── input ─────────────────────────────────────────────────────────────---
    col_input, col_btn = st.columns([6, 1])
    user_text = col_input.text_input(
        "Hypothesis (a single scientific claim)",
        placeholder="e.g. Scaling alone, without architectural change, is sufficient for general intelligence.",
    )
    go_btn = col_btn.button("Discover", type="primary")

    if go_btn and user_text:
        try:
            cleaned = normalise_hypothesis(user_text)
        except ValueError as e:
            st.error(str(e))
            return
        try:
            with st.spinner(
                f"Generating {st.session_state['n_perspectives']} perspectives, extracting "
                "activations on Lambda, and discovering axes…"
            ):
                disc = run_discovery(cleaned)
            st.session_state["discovery"] = disc
        except RuntimeError as e:
            st.error(
                f"Could not reach the Lambda GPU backend ({e}). "
                "Bring it up with `arguenaut-lambda up --wait-healthy`, then retry."
            )
            return
        except Exception as e:  # noqa: BLE001 — surface anything else to the UI
            st.exception(e)
            return

    disc: Discovery | None = st.session_state.get("discovery")
    if disc is None:
        st.info("Enter a hypothesis above and press **Discover** to begin.")
        return

    st.success(
        f"Discovered {len(disc.axes)} axes from {len(disc.perspectives)} perspectives "
        f"· model `{disc.model_id}` · layer {disc.layer}/{disc.n_layers}"
    )
    dropped = disc.n_dropped_judge + disc.n_dropped_duplicate + disc.n_dropped_outlier
    if disc.n_generated and dropped:
        st.caption(
            f"Quality filtering: generated {disc.n_generated}, dropped "
            f"{disc.n_dropped_judge} low-relevance · {disc.n_dropped_duplicate} duplicate · "
            f"{disc.n_dropped_outlier} outlier → kept {len(disc.perspectives)}."
        )

    # ── discovered axes ──────────────────────────────────────────────────────
    st.subheader("Discovered axes of disagreement")
    for ax in disc.axes:
        with st.container(border=True):
            st.markdown(_axis_caption(ax))
            if ax.outlier_driven:
                st.warning(
                    "⚠ This axis's variance is dominated by one or two perspectives — "
                    "treat it as a possible outlier artifact, not a robust shared axis."
                )
            if ax.rationale:
                st.caption(ax.rationale)
            hi, lo = st.columns(2)
            hi.markdown(f"**▲ {ax.high_pole}**")
            for t in ax.high_examples[:3]:
                hi.caption(f"• {t}")
            lo.markdown(f"**▼ {ax.low_pole}**")
            for t in ax.low_examples[:3]:
                lo.caption(f"• {t}")

    # ── scatter ──────────────────────────────────────────────────────────────
    st.subheader("Perspective map")
    n_pc = len(disc.explained_variance_ratio)
    opts = [f"PC{i + 1}" for i in range(n_pc)]
    cx, cy = st.columns(2)
    x_idx = cx.selectbox("X axis", range(n_pc), format_func=lambda i: opts[i], index=0)
    y_idx = cy.selectbox(
        "Y axis", range(n_pc), format_func=lambda i: opts[i], index=min(1, n_pc - 1)
    )
    st.plotly_chart(render_scatter(disc, x_idx, y_idx), use_container_width=True)

    # ── explained variance ───────────────────────────────────────────────────
    with st.expander("Explained variance & raw perspectives"):
        ev = disc.explained_variance_ratio
        bar = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev))
        bar.update_layout(
            title="Variance explained by each component",
            yaxis_title="fraction", height=280,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        st.plotly_chart(bar, use_container_width=True)
        st.dataframe(
            pd.DataFrame([{"stance": p.stance, "text": p.text} for p in disc.perspectives]),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
