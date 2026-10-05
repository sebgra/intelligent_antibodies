"""Streamlit dashboard for the Intelligent Antibodies project.

Generation results shown on the "Results" tab are produced by
`mock_results.generate_candidates`, a stand-in for the real pipeline in
`intelligent_antibodies/modules/utils/inference.py` (VAE decoder sampling +
Siamese interaction scoring). The mock mirrors that pipeline's actual
mechanics -- 2-D latent sampling, a `temperature` knob (now also added to
`inference.generate_antibody_sequence`), and a probability-style interaction
score -- so swapping in the real call once weights are available only means
replacing that one function call below; the rest of the UI is unaffected.

The "Dataset & Model" tab is **not** mocked: it reads the real SAbDab tables
shipped in `data/` and the real training-curve figures in `plots/`.
"""
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import real_pipeline
import structure_prediction
from mock_results import (
    ALPHABET_SIZE,
    RESIDUE_CLASS_COLOR,
    RESIDUE_CLASS_ORDER,
    SS_COLOR,
    VECTOR_SIZE,
    amino_acid_composition,
    colorize_sequence_html,
    generate_candidates as generate_candidates_mock,
    predict_secondary_structure,
    schematic_backbone,
)

# --------------------------------------------------------------------------
# Palette (validated categorical / sequential / status colors) + chrome ink.
# Categorical slots are assigned by fixed order, never cycled; sequential
# magnitude encodings use a single blue hue ramp.
# --------------------------------------------------------------------------
BRAND = "#008080"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

CAT_BLUE = "#2a78d6"     # slot 1 - Lead candidate / "passed"
CAT_ORANGE = "#eb6834"   # slot 2 - High confidence / "antibody chains"
CAT_AQUA = "#1baf7a"     # slot 3 - Candidate / "antigen chains"

STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

TIER_COLOR = {
    "Lead candidate": CAT_BLUE,
    "High confidence": CAT_ORANGE,
    "Candidate": CAT_AQUA,
}
TIER_ORDER = ["Lead candidate", "High confidence", "Candidate"]

EXAMPLE_ANTIGEN = (
    "KVFGRCELAAAMKRHGLANYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRT"
    "PGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)

st.set_page_config(
    page_title="Intelligent Antibodies Dashboard",
    page_icon="🧬",
    layout="wide",
)

# --------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .stTabs [data-baseweb="tab-list"] {{
            justify-content: center;
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {{
            color: {BRAND};
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] p {{
            color: {INK_MUTED};
        }}
        div[data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {GRIDLINE};
            border-radius: 10px;
            padding: 12px 16px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {INK_SECONDARY};
        }}
        .demo-banner {{
            background: #fff7e8;
            border: 1px solid {STATUS_WARNING};
            border-radius: 8px;
            padding: 8px 14px;
            color: {INK_PRIMARY};
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}
        .real-banner {{
            background: #eef6f6;
            border: 1px solid {BRAND};
            border-radius: 8px;
            padding: 8px 14px;
            color: {INK_PRIMARY};
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }}
        .spotlight-card {{
            border: 1px solid {CAT_BLUE};
            border-radius: 10px;
            padding: 16px 20px;
            background: linear-gradient(180deg, rgba(42,120,214,0.06), rgba(42,120,214,0.01));
            margin-bottom: 1rem;
        }}
        .tier-pill {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _plotly_base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRIDLINE),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=GRIDLINE)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, linecolor=GRIDLINE)
    return fig


def _tier_pill_html(tier: str) -> str:
    return f'<span class="tier-pill" style="background:{TIER_COLOR[tier]};">{tier}</span>'


def render_3dmol_viewer(pdb_text: str, scale_to_100: bool, height: int = 460) -> None:
    """
    Render a real 3-D structure (from ESMFold) with 3Dmol.js, cartoon style,
    colored by per-residue pLDDT using AlphaFold's own confidence bands.
    """
    scale_factor = 100 if scale_to_100 else 1
    html = f"""
    <div id="viewer" style="height:{height}px; width:100%; position:relative;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.5.5/3Dmol-min.js"></script>
    <script>
      const pdbData = {json.dumps(pdb_text)};
      const scaleFactor = {scale_factor};
      let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
      viewer.addModel(pdbData, "pdb");
      viewer.setStyle({{}}, {{cartoon: {{colorfunc: function(atom) {{
        const p = atom.b * scaleFactor;
        if (p >= 90) return "#0053D6";
        if (p >= 70) return "#65CBF3";
        if (p >= 50) return "#FFDB13";
        return "#FF7D45";
      }}}}}});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height)


def _selected_candidate_ids(event) -> List[str]:
    """
    Extract candidate IDs out of a plotly box/lasso-selection event (as
    returned by/stored for a `st.plotly_chart(..., on_select="rerun")` with
    `key="latent_chart"`). Points from the "below threshold" trace carry an
    empty id and are ignored. Defensive about the exact event shape so it
    degrades to "nothing selected" rather than raising.
    """
    if not event:
        return []
    selection = event.get("selection", {}) if hasattr(event, "get") else {}
    points = selection.get("points", [])
    ids = []
    for p in points:
        customdata = p.get("customdata") if hasattr(p, "get") else None
        if customdata and customdata[0]:
            ids.append(customdata[0])
    return sorted(set(ids))


@st.cache_data(show_spinner=False)
def load_sabdab_stats():
    """Load and summarize the real SAbDab tables shipped in data/SAbDab."""
    seq_path = REPO_ROOT / "data" / "SAbDab" / "sequences.csv"
    match_path = REPO_ROOT / "data" / "SAbDab" / "data_filtered.csv"

    seq_df = pd.read_csv(seq_path, sep=";")
    seq_df[["seq_rcpb", "seq_type"]] = seq_df["seq_id"].str.split("|", n=1, expand=True)
    seq_df["length"] = seq_df["sequence"].str.len()

    match_df = pd.read_csv(match_path, sep=";")

    species_counts = (
        seq_df["specie"].value_counts().head(10).rename_axis("species").reset_index(name="count")
    )

    return {
        "seq_df": seq_df,
        "match_df": match_df,
        "species_counts": species_counts,
    }


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
col_logo, col_title = st.columns([0.08, 0.92], gap="small", vertical_alignment="center")
with col_logo:
    logo_path = APP_DIR / "graphics" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=72)
with col_title:
    st.markdown(
        f"<h2 style='color:{BRAND}; margin-bottom:0;'>Intelligent Antibodies</h2>"
        f"<p style='color:{INK_SECONDARY}; margin-top:2px;'>"
        "In-silico antibody generation &amp; interaction screening</p>",
        unsafe_allow_html=True,
    )

tab_generate, tab_results, tab_data, tab_about = st.tabs(
    ["🧬 Generate", "📊 Results", "📁 Dataset & Model", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Generate tab
# --------------------------------------------------------------------------
with tab_generate:
    with st.container(border=True):
        st.subheader("Target antigen")
        antigen_input = st.text_area(
            label="Antigen sequence",
            placeholder="Enter an amino-acid sequence, e.g. " + EXAMPLE_ANTIGEN[:24] + "...",
            height=100,
        )
        use_example = st.checkbox("Use bundled example antigen (lysozyme, 1A2Y chain C)")
        st.caption("...or upload a FASTA file (first record is used).")
        uploaded_file = st.file_uploader(
            key="antigen_fasta_uploader", label="Choose a FASTA file", accept_multiple_files=False
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_requested = st.slider("Candidates to return", min_value=5, max_value=50, value=20, step=5)
        with col_b:
            threshold = st.slider(
                "Interaction score threshold", min_value=0.50, max_value=0.99, value=0.80, step=0.01,
                help="Same threshold as inference.test_interaction: a candidate is kept only if "
                     "the Siamese classifier's sigmoid score exceeds this value.",
            )
        with col_c:
            vector_size = int(st.number_input(
                "Sequence length (aa)", min_value=20, max_value=1000,
                value=VECTOR_SIZE, step=10,
                help="The one-hot encoding window (`vector_size`): the length of the generated "
                     "sequences and the input width of both models. Weights are stored per "
                     "length, so changing this loads the model trained for that length "
                     "(train one with `--vector-size <length>`); lengths without trained "
                     "weights fall back to simulated results.",
            ))

        temperature = st.slider(
            "Sampling temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1,
            help="Scales the latent-space sampling spread: z = temperature × N(0, 1), the exact "
                 "parameter added to generate_antibody_sequence() in modules/utils/inference.py. "
                 "Low temperature → conservative sampling near the VAE's prior (higher decode "
                 "fidelity, less diversity). High temperature → broader exploration of the latent "
                 "space (more diverse sequences, lower decode confidence).",
        )
        if temperature < 0.7:
            st.caption("🔵 Conservative sampling — candidates stay close to the training prior.")
        elif temperature > 1.4:
            st.caption("🟠 Exploratory sampling — expect more diverse but less confident decodes.")

        with st.expander("Encoding settings"):
            st.caption(
                f"One-hot encoding · {vector_size} aa window · {ALPHABET_SIZE}-letter alphabet · "
                "VAE latent dim 2 · Siamese CNN+GRU classifier "
                "(see intelligent_antibodies/modules/models/)"
            )

        use_real = real_pipeline.models_available(vector_size)
        if use_real:
            st.success(
                f"✅ Trained models for a {vector_size} aa window found under `run/models/` — "
                "generation will use them."
            )
        else:
            trained = real_pipeline.trained_vector_sizes()
            hint = (
                f" Trained weights exist for {', '.join(f'{s} aa' for s in trained)} — set the "
                "sequence length to one of those to run the real models."
                if trained else
                " Run `scripts/train_vae.py` and `scripts/train_siamese.py` to train real ones "
                "(see README.md, \"Training the models\")."
            )
            st.info(
                f"ℹ️ No trained models for a {vector_size} aa window under `run/models/` — "
                "showing **simulated** results." + hint
            )

        launch = st.button("🚀 Launch generation", type="primary")

    if launch:
        sequence = ""
        if uploaded_file is not None:
            raw = uploaded_file.read().decode("utf-8", errors="ignore")
            sequence = "".join(
                line.strip() for line in raw.splitlines() if line and not line.startswith(">")
            )
        elif use_example:
            sequence = EXAMPLE_ANTIGEN
        elif antigen_input:
            sequence = antigen_input.strip().upper()

        if not sequence:
            st.warning("Please enter a sequence, check the example box, or upload a FASTA file.")
        else:
            progress = st.progress(0, text="Encoding antigen sequence...")
            steps = [
                (20, "Encoding antigen sequence..."),
                (45, f"Sampling candidate antibodies (temperature={temperature:.1f})..."),
                (75, "Scoring antibody-antigen interactions..."),
                (100, "Ranking candidates..."),
            ]
            for pct, label in steps:
                time.sleep(0.25 if not use_real else 0.05)
                progress.progress(pct, text=label)
            if use_real:
                run = real_pipeline.generate_candidates_real(
                    sequence, n_requested=n_requested, threshold=threshold,
                    temperature=temperature, vector_size=vector_size,
                )
            else:
                run = generate_candidates_mock(
                    sequence, n_requested=n_requested, threshold=threshold,
                    temperature=temperature, vector_size=vector_size,
                )
            progress.empty()
            st.session_state["run"] = run
            st.success(
                f"Generated {run.n_passed} candidate(s) above threshold "
                f"from a pool of {run.n_pool_generated}. See the **Results** tab."
            )

# --------------------------------------------------------------------------
# Results tab
# --------------------------------------------------------------------------
with tab_results:
    run = st.session_state.get("run")

    if run is None:
        st.info("No results yet. Head to the **Generate** tab and launch a run.")
    else:
        if run.model_tag.startswith("real"):
            st.markdown(
                f'<div class="real-banner">✅ Generated with your trained models '
                f"(<code>{run.model_tag}</code>). Scores reflect however well those weights were "
                "actually trained — they are not a pretrained, production-grade model.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="demo-banner">⚠️ Simulated results — no trained models were found under '
                "<code>run/models/</code>, so these are randomly generated for interface development "
                "only. Train real ones with <code>scripts/train_vae.py</code> / "
                "<code>scripts/train_siamese.py</code> (see README.md).</div>",
                unsafe_allow_html=True,
            )

        meta_cols = st.columns(6)
        meta_cols[0].caption(f"**Antigen length**  \n{len(run.antigen_sequence)} aa")
        meta_cols[1].caption(f"**Candidates requested**  \n{run.n_requested}")
        meta_cols[2].caption(f"**Sequence length**  \n{run.vector_size} aa window")
        meta_cols[3].caption(f"**Pool sampled**  \n{run.n_pool_generated}")
        meta_cols[4].caption(f"**Temperature**  \n{run.temperature:.1f}")
        meta_cols[5].caption(f"**Generation time**  \n{run.elapsed_seconds:.1f} s")

        df = run.candidates
        pool = run.pool

        if df.empty:
            st.warning(
                "No candidates cleared the interaction threshold. Try lowering the "
                "threshold, raising the temperature, or requesting more candidates "
                "in the Generate tab."
            )
        else:
            sub_overview, sub_latent, sub_seqs, sub_explorer = st.tabs(
                ["Overview", "Latent space", "Sequence analysis", "Candidate explorer"]
            )

            # ================= Overview =================
            with sub_overview:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Candidates found", len(df))
                k2.metric("Top interaction score", f"{df['interaction_score'].max():.2f}")
                k3.metric("Mean decode confidence", f"{df['decode_confidence'].mean():.2f}")
                k4.metric("Pass rate (pool)", f"{100 * len(df) / len(pool):.1f} %")

                top = df.iloc[0]
                st.markdown(
                    f"""
                    <div class="spotlight-card">
                        <b>🏆 Top candidate — {top['candidate_id']}</b>
                        {_tier_pill_html(top['tier'])}
                        &nbsp;&nbsp;score {top['interaction_score']:.2f} ·
                        decode confidence {top['decode_confidence']:.2f} ·
                        latent z = ({top['z1']:.2f}, {top['z2']:.2f}) ·
                        length {top['length']} aa
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(top["sequence"], language=None)

                st.divider()
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("**Interaction score distribution across the sampled pool**")
                    fig_hist = go.Figure(
                        go.Histogram(
                            x=pool["interaction_score"],
                            nbinsx=30,
                            marker=dict(color="#5598e7", line=dict(color=SURFACE, width=1)),
                            hovertemplate="score %{x:.2f}<br>count %{y}<extra></extra>",
                        )
                    )
                    fig_hist.add_vline(
                        x=run.threshold, line=dict(color=STATUS_CRITICAL, dash="dash", width=2),
                        annotation_text=f"threshold {run.threshold:.2f}", annotation_position="top right",
                    )
                    fig_hist.update_layout(xaxis_title="Interaction score", yaxis_title="Candidates in pool")
                    st.plotly_chart(_plotly_base_layout(fig_hist), width="stretch")

                with chart_col2:
                    st.markdown("**Decode confidence vs. interaction score**")
                    fig_scatter = go.Figure()
                    for tier in TIER_ORDER:
                        sub = df[df["tier"] == tier]
                        if sub.empty:
                            continue
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=sub["interaction_score"],
                                y=sub["decode_confidence"],
                                mode="markers",
                                name=tier,
                                marker=dict(size=10, color=TIER_COLOR[tier], line=dict(width=1, color="white")),
                                customdata=sub[["candidate_id"]],
                                hovertemplate="%{customdata[0]}<br>score %{x:.2f}<br>confidence %{y:.2f}<extra></extra>",
                            )
                        )
                    fig_scatter.update_layout(
                        xaxis_title="Interaction score", yaxis_title="Decode confidence",
                        xaxis_range=[0, 1.02], yaxis_range=[0, 1.02],
                    )
                    st.plotly_chart(_plotly_base_layout(fig_scatter), width="stretch")

                st.markdown("**Top candidates by interaction score**")
                topN = df.head(min(10, len(df))).sort_values("interaction_score")
                fig_bar = go.Figure(
                    go.Bar(
                        x=topN["interaction_score"],
                        y=topN["candidate_id"],
                        orientation="h",
                        marker=dict(color="#2a78d6"),
                        text=[f"{v:.2f}" for v in topN["interaction_score"]],
                        textposition="outside",
                        hovertemplate="%{y}<br>score %{x:.2f}<extra></extra>",
                    )
                )
                fig_bar.update_layout(xaxis_title="Interaction score", yaxis_title="", xaxis_range=[0, 1.05])
                st.plotly_chart(
                    _plotly_base_layout(fig_bar, height=max(260, 28 * len(topN))), width="stretch"
                )

                st.divider()
                st.markdown("**All candidates**")
                filter_cols = st.columns([0.3, 0.4, 0.3])
                with filter_cols[0]:
                    tier_filter = st.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
                with filter_cols[1]:
                    min_score = st.slider("Minimum score to display", 0.0, 1.0, run.threshold, 0.01)
                with filter_cols[2]:
                    min_conf = st.slider("Minimum decode confidence", 0.0, 1.0, 0.0, 0.05)

                table_df = df[
                    df["tier"].isin(tier_filter)
                    & (df["interaction_score"] >= min_score)
                    & (df["decode_confidence"] >= min_conf)
                ]

                st.dataframe(
                    table_df[[
                        "rank", "candidate_id", "tier", "sequence", "length",
                        "interaction_score", "decode_confidence", "z1", "z2",
                    ]],
                    column_config={
                        "rank": "Rank",
                        "candidate_id": "ID",
                        "tier": "Tier",
                        "sequence": st.column_config.TextColumn("Sequence", width="large"),
                        "length": st.column_config.NumberColumn("Length (aa)"),
                        "interaction_score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=1, format="%.2f"
                        ),
                        "decode_confidence": st.column_config.ProgressColumn(
                            "Decode confidence", min_value=0, max_value=1, format="%.2f"
                        ),
                        "z1": st.column_config.NumberColumn("Latent z₁", format="%.2f"),
                        "z2": st.column_config.NumberColumn("Latent z₂", format="%.2f"),
                    },
                    hide_index=True,
                    width="stretch",
                )

                st.download_button(
                    "⬇️ Download results as CSV",
                    data=df.drop(columns="decode_trace").to_csv(index=False).encode("utf-8"),
                    file_name="intelligent_antibodies_candidates.csv",
                    mime="text/csv",
                )

            # ================= Latent space =================
            with sub_latent:
                st.markdown(
                    f"**Sampled points in the VAE's 2-D latent space** "
                    f"(temperature = {run.temperature:.1f})"
                )
                st.caption(
                    "Every point is one latent draw z = temperature × N(0, 1), decoded into a "
                    "candidate sequence. Dashed circles mark the 1σ/2σ bounds of the *untempered* "
                    "prior (radius = temperature, 2×temperature) — points landing outside the "
                    "inner circle at temperature > 1 are the 'exploratory' samples enabled by a "
                    "higher temperature. **Drag a box or lasso over the candidate points** (the "
                    "colored ones) to filter the Sequence analysis and Candidate explorer tabs "
                    "down to just that selection."
                )

                fig_latent = go.Figure()
                theta = np.linspace(0, 2 * np.pi, 100)
                for radius, dash in [(run.temperature, "dot"), (2 * run.temperature, "dash")]:
                    fig_latent.add_trace(
                        go.Scatter(
                            x=radius * np.cos(theta), y=radius * np.sin(theta),
                            mode="lines", line=dict(color=INK_MUTED, width=1, dash=dash),
                            showlegend=False, hoverinfo="skip",
                        )
                    )
                failed = pool[~pool["sequence"].isin(df["sequence"])].copy()
                failed["candidate_id"] = ""  # not a real candidate -- excluded from selection
                fig_latent.add_trace(
                    go.Scatter(
                        x=failed["z1"], y=failed["z2"], mode="markers", name="Below threshold",
                        marker=dict(size=7, color=GRIDLINE, line=dict(width=1, color=INK_MUTED)),
                        customdata=failed[["candidate_id", "interaction_score"]],
                        hovertemplate="score %{customdata[1]:.2f}<extra></extra>",
                    )
                )
                for tier in TIER_ORDER:
                    sub = df[df["tier"] == tier]
                    if sub.empty:
                        continue
                    fig_latent.add_trace(
                        go.Scatter(
                            x=sub["z1"], y=sub["z2"], mode="markers", name=tier,
                            marker=dict(size=11, color=TIER_COLOR[tier], line=dict(width=1, color="white")),
                            customdata=sub[["candidate_id", "interaction_score"]],
                            hovertemplate="%{customdata[0]}<br>score %{customdata[1]:.2f}<extra></extra>",
                        )
                    )
                fig_latent.add_trace(
                    go.Scatter(
                        x=[0], y=[0], mode="markers", name="Prior mean",
                        marker=dict(size=10, color=INK_PRIMARY, symbol="x"), hoverinfo="skip",
                    )
                )
                fig_latent.update_layout(xaxis_title="z₁", yaxis_title="z₂", dragmode="lasso")
                fig_latent.update_yaxes(scaleanchor="x", scaleratio=1)
                latent_event = st.plotly_chart(
                    _plotly_base_layout(fig_latent, height=480), width="stretch",
                    key="latent_chart", on_select="rerun", selection_mode=["box", "lasso"],
                )

                selected_ids = _selected_candidate_ids(latent_event)
                shown_df = df[df["candidate_id"].isin(selected_ids)] if selected_ids else df

                sel_col1, sel_col2 = st.columns([0.8, 0.2])
                with sel_col1:
                    if selected_ids:
                        st.success(f"🔎 {len(selected_ids)} candidate(s) selected: " + ", ".join(selected_ids))
                    else:
                        st.caption("No selection — Sequence analysis and Candidate explorer show all candidates.")
                with sel_col2:
                    if selected_ids and st.button("Clear selection"):
                        del st.session_state["latent_chart"]
                        st.rerun()

                lc1, lc2 = st.columns(2)
                observed_spread = float(np.linalg.norm(pool[["z1", "z2"]].values, axis=1).mean())
                expected_spread = run.temperature * np.sqrt(np.pi / 2)
                lc1.metric("Observed mean ‖z‖", f"{observed_spread:.2f}")
                lc2.metric("Expected mean ‖z‖ at this T", f"{expected_spread:.2f}")

            # ================= Sequence analysis =================
            with sub_seqs:
                if selected_ids:
                    st.success(f"Showing stats for the {len(shown_df)} candidate(s) selected in the Latent space tab.")
                else:
                    st.caption(f"Showing stats for all {len(shown_df)} generated candidates. "
                               "Box/lasso-select points in the Latent space tab to narrow this down.")

                sa1, sa2, sa3, sa4 = st.columns(4)
                sa1.metric("Candidates shown", len(shown_df))
                sa2.metric("Mean length", f"{shown_df['length'].mean():.0f} aa")
                sa3.metric("Mean hydrophobicity (KD)", f"{shown_df['hydrophobicity'].mean():.2f}")
                sa4.metric("Mean net charge", f"{shown_df['net_charge'].mean():+.1f}")

                seq_col1, seq_col2 = st.columns(2)
                with seq_col1:
                    st.markdown("**Length distribution**")
                    fig_sa_len = go.Figure(
                        go.Histogram(
                            x=shown_df["length"], nbinsx=max(5, min(20, len(shown_df))),
                            marker=dict(color="#5598e7", line=dict(color=SURFACE, width=1)),
                            hovertemplate="length %{x}<br>count %{y}<extra></extra>",
                        )
                    )
                    fig_sa_len.update_layout(xaxis_title="Length (aa)", yaxis_title="Candidates")
                    st.plotly_chart(_plotly_base_layout(fig_sa_len, height=320), width="stretch")

                with seq_col2:
                    st.markdown("**Hydrophobicity vs. net charge**")
                    fig_sa_pc = go.Figure()
                    for tier in TIER_ORDER:
                        sub = shown_df[shown_df["tier"] == tier]
                        if sub.empty:
                            continue
                        fig_sa_pc.add_trace(
                            go.Scatter(
                                x=sub["net_charge"], y=sub["hydrophobicity"], mode="markers", name=tier,
                                marker=dict(size=10, color=TIER_COLOR[tier], line=dict(width=1, color="white")),
                                customdata=sub[["candidate_id"]],
                                hovertemplate="%{customdata[0]}<br>charge %{x:+.0f}<br>hydrophobicity %{y:.2f}<extra></extra>",
                            )
                        )
                    fig_sa_pc.add_vline(x=0, line=dict(color=GRIDLINE, width=1))
                    fig_sa_pc.update_layout(xaxis_title="Net charge (≈ at pH 7)", yaxis_title="Mean hydrophobicity (Kyte-Doolittle)")
                    st.plotly_chart(_plotly_base_layout(fig_sa_pc, height=320), width="stretch")

                st.markdown("**Amino-acid composition — shown selection vs. full candidate set**")
                shown_comp = pd.concat([amino_acid_composition(s) for s in shown_df["sequence"]])
                shown_comp = shown_comp.groupby("residue")["pct"].mean()
                all_comp = pd.concat([amino_acid_composition(s) for s in df["sequence"]])
                all_comp = all_comp.groupby("residue")["pct"].mean()
                fig_sa_comp = go.Figure()
                fig_sa_comp.add_trace(go.Bar(
                    x=list(all_comp.index), y=all_comp.values, name="All candidates",
                    marker=dict(color=GRIDLINE),
                ))
                fig_sa_comp.add_trace(go.Bar(
                    x=list(shown_comp.index), y=shown_comp.values, name="Shown selection",
                    marker=dict(color="#2a78d6"),
                ))
                fig_sa_comp.update_layout(
                    xaxis_title="Residue", yaxis_title="Mean frequency (%)", barmode="group",
                )
                st.plotly_chart(_plotly_base_layout(fig_sa_comp, height=320), width="stretch")

                st.divider()
                st.markdown("**Colorized sequences**")
                legend_html = " &nbsp; ".join(
                    f'<span style="color:{RESIDUE_CLASS_COLOR[cls]};font-weight:600;">■</span> {cls}'
                    for cls in RESIDUE_CLASS_ORDER
                )
                st.markdown(legend_html, unsafe_allow_html=True)
                for _, row in shown_df.head(15).iterrows():
                    st.caption(f"{row['candidate_id']} · score {row['interaction_score']:.2f} · {row['length']} aa")
                    st.markdown(colorize_sequence_html(row["sequence"]), unsafe_allow_html=True)
                if len(shown_df) > 15:
                    st.caption(f"...and {len(shown_df) - 15} more. Narrow the selection in the Latent space tab to see others.")

            # ================= Candidate explorer =================
            with sub_explorer:
                st.caption(
                    f"Choosing from {'the ' + str(len(shown_df)) + ' selected' if selected_ids else 'all ' + str(len(shown_df))} "
                    "candidate(s). Select points in the Latent space tab to narrow this list."
                )
                selected_id = st.selectbox("Select a candidate", shown_df["candidate_id"])
                selected = df[df["candidate_id"] == selected_id].iloc[0]

                detail_col1, detail_col2 = st.columns([0.55, 0.45])
                with detail_col1:
                    st.caption(
                        f"{selected['candidate_id']} · {selected['tier']} · "
                        f"{selected['length']} aa · z = ({selected['z1']:.2f}, {selected['z2']:.2f})"
                    )
                    st.code(selected["sequence"], language=None)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Interaction score", f"{selected['interaction_score']:.2f}")
                    m2.metric("Decode confidence", f"{selected['decode_confidence']:.2f}")
                    m3.metric("Hydrophobicity", f"{selected['hydrophobicity']:.2f}")

                    st.markdown("**Colorized sequence** (by physicochemical class)")
                    legend_html = " &nbsp; ".join(
                        f'<span style="color:{RESIDUE_CLASS_COLOR[cls]};font-weight:600;">■</span> {cls}'
                        for cls in RESIDUE_CLASS_ORDER
                    )
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.markdown(colorize_sequence_html(selected["sequence"]), unsafe_allow_html=True)

                    st.markdown("**Per-residue decode confidence**")
                    trace = selected["decode_trace"]
                    fig_trace = go.Figure(
                        go.Scatter(
                            x=list(range(1, len(trace) + 1)), y=trace, mode="lines",
                            line=dict(color="#2a78d6", width=2),
                            hovertemplate="position %{x}<br>confidence %{y:.2f}<extra></extra>",
                        )
                    )
                    fig_trace.add_hline(y=0.5, line=dict(color=GRIDLINE, dash="dot"))
                    fig_trace.update_layout(xaxis_title="Residue position", yaxis_title="Confidence")
                    st.plotly_chart(_plotly_base_layout(fig_trace, height=240), width="stretch")

                with detail_col2:
                    comp = amino_acid_composition(selected["sequence"]).sort_values("pct", ascending=True)
                    fig_comp = go.Figure(
                        go.Bar(
                            x=comp["pct"], y=comp["residue"], orientation="h",
                            marker=dict(color="#2a78d6"),
                            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                        )
                    )
                    fig_comp.update_layout(xaxis_title="Residue frequency (%)", yaxis_title="")
                    st.plotly_chart(_plotly_base_layout(fig_comp, height=420), width="stretch")

                st.divider()
                st.markdown("**Folded representation (schematic)**")
                st.caption(
                    "⚠️ Illustrative only — this is a simplified Chou-Fasman-style secondary-structure "
                    "call rendered as a toy 3-D backbone (helices spiral, strands zigzag, coils wander). "
                    "It is **not** a real structure prediction: no AlphaFold/ESMFold model is connected. "
                    "Treat it as a placeholder for where a real fold viewer will go once one is wired in."
                )
                ss = predict_secondary_structure(selected["sequence"])
                backbone = schematic_backbone(selected["sequence"], ss, seed=hash(selected["candidate_id"]) % (2**32))

                fig_fold = go.Figure()
                fig_fold.add_trace(go.Scatter3d(
                    x=backbone["x"], y=backbone["y"], z=backbone["z"], mode="lines",
                    line=dict(color=INK_MUTED, width=3), showlegend=False, hoverinfo="skip",
                ))
                for ss_name, color in SS_COLOR.items():
                    sub_bb = backbone[backbone["ss_label"] == ss_name]
                    if sub_bb.empty:
                        continue
                    fig_fold.add_trace(go.Scatter3d(
                        x=sub_bb["x"], y=sub_bb["y"], z=sub_bb["z"], mode="markers", name=ss_name,
                        marker=dict(size=4, color=color),
                        customdata=sub_bb[["position", "residue"]],
                        hovertemplate="residue %{customdata[1]} (pos %{customdata[0]})<br>" + ss_name + "<extra></extra>",
                    ))
                fig_fold.update_layout(
                    height=480,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY, size=13),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
                    scene=dict(
                        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                        aspectmode="data",
                    ),
                )
                st.plotly_chart(fig_fold, width="stretch")
                ss_counts = pd.Series(ss).value_counts()
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Helix", f"{100 * ss_counts.get('H', 0) / len(ss):.0f}%")
                fc2.metric("Sheet", f"{100 * ss_counts.get('E', 0) / len(ss):.0f}%")
                fc3.metric("Coil", f"{100 * ss_counts.get('C', 0) / len(ss):.0f}%")

                st.divider()
                st.markdown("**Real ML folding — ESMFold** 🧬")
                st.caption(
                    "Full AlphaFold2 needs a multi-gigabyte sequence-database search and a GPU "
                    "cluster, neither available here. **ESMFold** is a real, single-sequence "
                    "structure predictor from the same line of research, often used as a practical "
                    "AlphaFold stand-in — this calls the free public ESM Atlas API over the "
                    "internet, so it needs a connection and can take up to ~a minute."
                )
                fold_key = f"fold_{selected['candidate_id']}"
                if st.button("🔬 Fold with ESMFold", key=f"btn_{fold_key}"):
                    st.session_state[fold_key] = "requested"

                if st.session_state.get(fold_key) == "requested":
                    try:
                        with st.spinner("Calling the ESMFold API (this can take ~30-60s)..."):
                            fold_result = structure_prediction.fold_sequence(selected["sequence"])
                    except structure_prediction.FoldingError as exc:
                        st.error(f"Could not get a real fold: {exc}")
                    else:
                        st.session_state[fold_key] = fold_result

                fold_state = st.session_state.get(fold_key)
                if isinstance(fold_state, structure_prediction.FoldResult):
                    fold_result = fold_state
                    render_3dmol_viewer(fold_result.pdb_text, scale_to_100=fold_result.raw_bfactor_is_0_to_1)
                    plddt_legend = " &nbsp; ".join(
                        f'<span style="color:{color};font-weight:600;">■</span> {label} ({lo}-{hi})'
                        for lo, hi, color, label in structure_prediction.PLDDT_BANDS
                    )
                    st.markdown(
                        f"Per-residue confidence (pLDDT) — AlphaFold's own color convention: "
                        f"{plddt_legend}",
                        unsafe_allow_html=True,
                    )
                    plddt_cols = st.columns(4)
                    plddt_cols[0].metric("Mean pLDDT", f"{fold_result.mean_plddt:.1f}")
                    plddt_cols[1].metric("Min pLDDT", f"{min(fold_result.per_residue_plddt):.1f}")
                    plddt_cols[2].metric("Max pLDDT", f"{max(fold_result.per_residue_plddt):.1f}")
                    plddt_cols[3].metric("Residues folded", len(fold_result.per_residue_plddt))

        if st.button("🗑️ Clear results"):
            del st.session_state["run"]
            st.rerun()

# --------------------------------------------------------------------------
# Dataset & Model tab (real data, real artifacts -- nothing mocked here)
# --------------------------------------------------------------------------
with tab_data:
    st.markdown(
        '<div class="real-banner">✅ Everything on this tab is computed from the real data files '
        "in <code>data/SAbDab/</code> and the real training figures in <code>plots/</code> — "
        "none of it is simulated.</div>",
        unsafe_allow_html=True,
    )

    try:
        stats = load_sabdab_stats()
    except FileNotFoundError as exc:
        st.warning(f"Could not load SAbDab tables: {exc}")
        stats = None

    if stats is not None:
        seq_df = stats["seq_df"]
        match_df = stats["match_df"]

        st.subheader("SAbDab dataset")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Sequences", len(seq_df))
        d2.metric("Antibody chains", int((seq_df["seq_type"] == "ab").sum()))
        d3.metric("Antigen chains", int((seq_df["seq_type"] == "ag").sum()))
        d4.metric("Labeled ab/ag pairs", len(match_df))

        over_cutoff = (seq_df["length"] > vector_size).mean() * 100

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**Sequence length distribution**")
            fig_len = go.Figure(
                go.Histogram(
                    x=seq_df["length"], nbinsx=40,
                    marker=dict(color="#5598e7", line=dict(color=SURFACE, width=1)),
                    hovertemplate="length %{x}<br>count %{y}<extra></extra>",
                )
            )
            fig_len.add_vline(
                x=vector_size, line=dict(color=STATUS_CRITICAL, dash="dash", width=2),
                annotation_text=f"model window = {vector_size} aa", annotation_position="top right",
            )
            fig_len.update_layout(xaxis_title="Sequence length (aa)", yaxis_title="Count")
            st.plotly_chart(_plotly_base_layout(fig_len), width="stretch")
            st.caption(
                f"{over_cutoff:.1f}% of sequences exceed the model's {vector_size} aa encoding "
                "window and are truncated by `ProteinOneHotEncoder`."
            )

        with dcol2:
            st.markdown("**Class balance (raw, before equilibration)**")
            balance = match_df["interaction"].value_counts().rename({0: "Non-interacting", 1: "Interacting"})
            fig_balance = go.Figure(
                go.Bar(
                    x=balance.index, y=balance.values,
                    marker=dict(color=[CAT_ORANGE, CAT_BLUE]),
                    text=[f"{v:,}" for v in balance.values], textposition="outside",
                )
            )
            fig_balance.update_layout(xaxis_title="", yaxis_title="Pairs", yaxis_type="log")
            st.plotly_chart(_plotly_base_layout(fig_balance), width="stretch")
            st.caption(
                f"Only {100 * balance.get('Interacting', 0) / balance.sum():.2f}% of labeled pairs "
                "interact — this is why `dataset.EquilibratedDataset` downsamples the negative "
                "class to match the positive class before training."
            )

        st.markdown("**Top 10 species represented**")
        species_counts = stats["species_counts"].sort_values("count")
        fig_species = go.Figure(
            go.Bar(
                x=species_counts["count"], y=species_counts["species"], orientation="h",
                marker=dict(color="#2a78d6"),
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        fig_species.update_layout(xaxis_title="Sequences", yaxis_title="")
        st.plotly_chart(_plotly_base_layout(fig_species, height=380), width="stretch")
        st.caption("Source: `data/SAbDab/sequences.csv` / `data_filtered.csv` (see `scripts/get_seq_table.py`, "
                    "`scripts/get_interaction_table.py`).")

    st.divider()
    st.subheader("Model architecture")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("**Generator — Convolutional VAE** (`models/VAEFull.py`)")
        st.caption(
            f"Conv2D(32) → Conv2D(64), stride 2 each → Dense(16) → "
            f"z_mean / z_log_var → latent dim **{2}** → "
            "Dense → Reshape → Conv2DTranspose ×2 → sigmoid reconstruction. "
            f"Operates on a ({vector_size} × {ALPHABET_SIZE} × 1) one-hot image per sequence."
        )
    with arch_col2:
        st.markdown("**Discriminator — Siamese classifier** (`models/SiameseInteractionClassifier.py`)")
        st.caption(
            "Shared tower: 4×(Conv1D + MaxPool1D) with increasing filter widths, "
            "then a Bidirectional GRU. Antibody and antigen embeddings are multiplied "
            "element-wise and passed through Dropout(0.2) → Dense(1, sigmoid)."
        )

    st.markdown("**Training metrics** (`models/SiameseInteractionClassifier.py`)")
    st.caption(
        "accuracy = (TP+TN)/(TP+TN+FP+FN) · "
        "F1 = 2·precision·recall/(precision+recall) · "
        "MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))"
    )

    plot_files = {
        "Siamese — training curve": "Siamese_training_curve.png",
        "Siamese — training metrics": "Siamese_training_metrics.png",
        "Siamese — training (early run)": "Siamese_training.png",
        "VAE — training curve": "VAE_training_curve.png",
    }
    existing_plots = {
        label: REPO_ROOT / "plots" / fname
        for label, fname in plot_files.items()
        if (REPO_ROOT / "plots" / fname).exists()
    }
    if existing_plots:
        st.markdown("**Training diagnostics** (captured from prior runs, see `plots/`)")
        plot_cols = st.columns(2)
        for i, (label, path) in enumerate(existing_plots.items()):
            with plot_cols[i % 2]:
                st.image(str(path), caption=label, width="stretch")

# --------------------------------------------------------------------------
# About tab
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
Therapeutic (monoclonal) antibodies are one of the most effective therapies available today for
the treatment of chronic inflammatory diseases such as Crohn's disease, lupus and multiple
sclerosis. To treat the latter, monoclonal antibodies can target certain proteins involved in
these pathologies with a view to neutralizing them, and can also be used to limit the supply of
factors essential to tumor growth or disruptors of the tumor microenvironment. Monoclonal
antibody-based serotherapy can also compensate for treatment shortfalls in the case of fulminant
epidemics where the pathogens involved have a high mutability rate, such as COVID-19.

Although promising and a major product on the pharmaceutical market, only around thirty
monoclonal antibodies are currently available for chronic inflammatory diseases, and around ten
for the treatment of cancer. This lack of comprehensiveness is due to the many difficulties
inherent in the in-vitro and in-silico design of these therapeutic molecules. Antibody design
and/or optimization remains a real challenge, not least because of the need to produce molecules
that are effective, target-specific and deliverable to the organs being treated. The difficulties
are also linked to long and costly development times.

In order to accelerate the development of therapeutic antibodies, in-silico methods have been
developed to reduce modeling times for these molecules, while exploring design possibilities
more exhaustively. Although advantageous, these methods currently rely essentially on estimating
the affinity between the antibody and its target by calculating the binding energy, which remains
difficult to estimate and extremely time-consuming from an experimental point of view.
        """
    )
    st.caption(
        "Pipeline: a convolutional VAE samples candidate antibody sequences from a learned "
        "latent space; a Siamese CNN+GRU classifier scores each candidate against the target "
        "antigen. See `intelligent_antibodies/modules/` for the model code and "
        "`scripts/` for the SAbDab/CoV-AbDab data-collection utilities."
    )
