from html import escape
from io import BytesIO

import streamlit as st

from predict import classify_image, load_predictor


st.set_page_config(
    page_title="Thread / Index",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_predictor():
    return load_predictor()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #182b27;
        --muted: #71817c;
        --paper: #f2f7f4;
        --line: #d8e4df;
        --green: #176b52;
        --lime: #c8f169;
        --coral: #e9755d;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(200, 241, 105, .18), transparent 34%),
            linear-gradient(145deg, #f7faf8 0%, var(--paper) 60%, #edf4f0 100%);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }

    .masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--line);
        padding: 0 0 1rem;
        margin-bottom: 3rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        font-size: .84rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: var(--green);
        color: white;
        font-size: .76rem;
    }
    .mast-meta {
        color: var(--muted);
        font: 500 .68rem 'DM Mono', monospace;
        letter-spacing: .05em;
        text-transform: uppercase;
    }
    .eyebrow {
        color: var(--green);
        font: 500 .7rem 'DM Mono', monospace;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .intro h1 {
        color: var(--ink);
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        font-weight: 700;
        line-height: 1.02;
        letter-spacing: -.055em;
        margin: 0;
    }
    .intro h1 span { color: var(--green); }
    .intro-copy {
        color: #586963;
        max-width: 520px;
        font-size: 1rem;
        line-height: 1.7;
        margin: 1.2rem 0 2.3rem;
    }
    .panel-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 .3rem;
    }
    .panel-note { color: var(--muted); font-size: .82rem; margin-bottom: 1.2rem; }
    .result-panel {
        background: rgba(255, 255, 255, .72);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.5rem;
        min-height: 245px;
    }
    .empty-mark {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: #e4eee8;
        color: var(--green);
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .result-label {
        color: var(--muted);
        font: 500 .68rem 'DM Mono', monospace;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 1.3rem 0 .35rem;
    }
    .result-category {
        color: var(--green);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        text-transform: capitalize;
        letter-spacing: -.04em;
    }
    .score-caption {
        color: var(--muted);
        font: 500 .7rem 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .confidence-line, .score-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        color: #43564f;
        font-size: .82rem;
        font-weight: 600;
    }
    .confidence-line { margin-top: .9rem; }
    .confidence-line strong {
        color: var(--green);
        font: 700 1rem 'DM Mono', monospace;
    }
    .score-list { display: grid; gap: .85rem; }
    .score-heading { margin-bottom: .35rem; }
    .score-heading strong {
        color: var(--ink);
        font: 700 .82rem 'DM Mono', monospace;
        white-space: nowrap;
    }
    .score-track {
        height: 10px;
        overflow: hidden;
        border-radius: 999px;
        background: #dce6e1;
    }
    .score-fill {
        height: 100%;
        border-radius: inherit;
        background: var(--green);
    }
    .footnote {
        border-top: 1px solid var(--line);
        margin-top: 2.3rem;
        padding-top: 1rem;
        color: var(--muted);
        font: 400 .67rem 'DM Mono', monospace;
        letter-spacing: .025em;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, .65);
        border: 1px dashed #a9bdb4;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] button {
        border: 1px solid var(--green);
        color: var(--green);
        border-radius: 5px;
        font-weight: 700;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--green);
        color: white;
        border-color: var(--green);
    }
    [data-testid="stProgressBar"] > div > div > div > div {
        background-color: var(--green);
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .masthead { margin-bottom: 2rem; }
        .mast-meta { display: none; }
        .intro h1 { font-size: 2.7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="masthead">
      <div class="brand"><span class="brand-mark">T</span> Thread / Index</div>
      <div class="mast-meta">Image classification &nbsp; / &nbsp; 5 categories</div>
    </div>
    <div class="intro">
      <div class="eyebrow">Wardrobe vision system &nbsp; 01</div>
      <h1>Give an image.<br><span>Find its category.</span></h1>
      <p class="intro-copy">Drop in a clothing photo and the model will identify its closest category.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, result_column = st.columns([1.05, 0.95], gap="large")

with upload_column:
    st.markdown('<div class="panel-title">01 / Upload image</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">JPG, JPEG, or PNG · one image at a time</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, width=700)
    else:
        st.markdown(
            '<div class="empty-mark">↥</div><div class="panel-note">Your image preview will appear here.</div>',
            unsafe_allow_html=True,
        )

with result_column:
    st.markdown('<div class="panel-title">02 / Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Prediction from the trained clothing model</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    if uploaded_file:
        try:
            with st.spinner("Analyzing image…"):
                prediction = classify_image(
                    BytesIO(uploaded_file.getvalue()),
                    *get_predictor(),
                )
            st.markdown(
                f'<div class="result-panel"><div class="result-label">Predicted category</div>'
                f'<div class="result-category">{escape(prediction["category"].title())}</div>'
                f'<div class="confidence-line"><span>Prediction confidence</span>'
                f'<strong>{prediction["confidence"] * 100:.1f}%</strong></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="result-label">Category scores</div>', unsafe_allow_html=True)
            score_rows = []
            for category, score in sorted(
                prediction["probabilities"].items(), key=lambda item: item[1], reverse=True
            ):
                percentage = max(0.0, min(float(score) * 100, 100.0))
                score_rows.append(
                    f'<div class="score-item"><div class="score-heading">'
                    f'<span>{escape(category.title())}</span><strong>{percentage:.1f}%</strong></div>'
                    f'<div class="score-track" role="progressbar" '
                    f'aria-label="{escape(category.title())} score" aria-valuemin="0" '
                    f'aria-valuemax="100" aria-valuenow="{percentage:.1f}">'
                    f'<div class="score-fill" style="width: {percentage:.1f}%"></div></div></div>'
                )
            st.markdown(f'<div class="score-list">{"".join(score_rows)}</div>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Model files are missing. Run `python train.py` to create them first.")
        except Exception as error:
            st.error(f"Could not classify this image: {error}")
    else:
        st.markdown(
            '<div class="result-panel"><div class="empty-mark">◎</div>'
            '<div class="panel-note">The predicted category and model scores will show up here.</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footnote">MOBILENETV2 FEATURE EXTRACTION &nbsp; · &nbsp; LINEAR SVM CLASSIFIER</div>',
    unsafe_allow_html=True,
)
