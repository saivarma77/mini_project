"""
AI Agent for Customer Review Sentiment Analysis & Business Intelligence Generation
------------------------------------------------------------------------------
A Streamlit application that:
  1. Ingests customer reviews (CSV upload or sample dataset)
  2. Runs sentiment classification using RoBERTa / DeBERTa / DistilBERT
  3. Extracts key aspects/topics customers talk about
  4. Generates an automated business intelligence report + interactive dashboard

Run with:
    streamlit run app.py
"""

import io
import re
import time
from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.text_utils import clean_text, extract_aspects, STOPWORDS
from utils.report_generator import generate_bi_report
from utils.model_loader import get_sentiment_pipeline, MODEL_OPTIONS

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Review Intelligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6b7280;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">📊 AI Agent for Review Sentiment Analysis & Business Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload customer reviews, classify sentiment with transformer models, and generate an automated BI report.</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "model_used" not in st.session_state:
    st.session_state.model_used = None
if "benchmark_df" not in st.session_state:
    st.session_state.benchmark_df = None

# --------------------------------------------------------------------------
# SIDEBAR - NAVIGATION & CONFIGURATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Navigation")
    selected_tab = st.radio(
        "Select View",
        ["📈 Dashboard", "🏷️ Aspect Analysis", "📝 BI Report", "⚖️ Model Comparison", "🗂️ Raw Data"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.header("⚙️ Configuration")

    data_source = st.radio(
        "Data source",
        ["Use sample dataset", "Upload CSV"],
        help="Sample dataset is a small demo set of Amazon-style reviews.",
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV with a 'review' text column",
            type=["csv"],
        )

    st.markdown("---")
    st.subheader("Model Selection")
    model_key = st.selectbox(
        "Sentiment model",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="Choose which transformer model to use for sentiment classification.",
    )
    st.caption(MODEL_OPTIONS[model_key]["description"])

    max_rows = st.slider(
        "Max reviews to process",
        min_value=10,
        max_value=2000,
        value=200,
        step=10,
        help="Limit processing for speed. Increase if you have GPU / more time.",
    )

    batch_size = st.select_slider(
        "Batch size", options=[4, 8, 16, 32], value=8
    )

    st.markdown("---")
    run_button = st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("Compare Models")
    compare_button = st.button("⚖️ Benchmark All 3 Models (sample of 50)", use_container_width=True)

    st.markdown("---")
    st.caption(
        "Models used: **RoBERTa**, **DeBERTa**, **DistilBERT** "
        "(via Hugging Face Transformers). First run downloads model weights "
        "and may take a few minutes."
    )

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
def load_data():
    if data_source == "Use sample dataset":
        df = pd.read_csv("data/sample_reviews.csv")
    else:
        if uploaded_file is None:
            return None
        df = pd.read_csv(uploaded_file)
        # try to find a text column
        text_col = None
        for c in df.columns:
            if c.lower() in ("review", "text", "review_text", "content", "comment"):
                text_col = c
                break
        if text_col is None:
            text_col = df.columns[0]
        df = df.rename(columns={text_col: "review"})
        if "review" not in df.columns:
            st.error("Could not find a review text column in the uploaded CSV.")
            return None
    return df


df_raw = load_data()

if df_raw is None:
    st.info("👈 Choose a data source in the sidebar to get started (sample dataset works out of the box).")
    st.stop()

df_raw = df_raw.head(max_rows).copy()
df_raw["review"] = df_raw["review"].astype(str)
df_raw["clean_review"] = df_raw["review"].apply(clean_text)

# add a synthetic date/category column if not present, useful for BI trends
if "date" not in df_raw.columns:
    df_raw["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_raw)).to_list()
if "category" not in df_raw.columns:
    demo_categories = ["Electronics", "Home & Kitchen", "Fashion", "Books", "Beauty", "Sports"]
    rng = np.random.default_rng(42)
    df_raw["category"] = rng.choice(demo_categories, size=len(df_raw))

# --------------------------------------------------------------------------
# RUN SENTIMENT ANALYSIS
# --------------------------------------------------------------------------
def run_analysis(df, model_key, batch_size):
    pipe = get_sentiment_pipeline(model_key)
    texts = df["clean_review"].tolist()

    progress = st.progress(0, text="Running sentiment inference...")
    labels, scores = [], []

    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        batch = [t if len(t.strip()) > 0 else "empty review" for t in batch]
        preds = pipe(batch, truncation=True)
        for p in preds:
            label = p["label"].upper()
            # normalize label formats across models (POSITIVE/NEGATIVE, LABEL_0/1, 1-5 stars, etc.)
            norm_label, norm_score = normalize_label(label, p["score"])
            labels.append(norm_label)
            scores.append(norm_score)
        progress.progress(min(1.0, (i + batch_size) / n), text=f"Processed {min(i+batch_size, n)}/{n} reviews")

    progress.empty()
    df = df.copy()
    df["sentiment"] = labels
    df["confidence"] = scores
    return df


def normalize_label(label, score):
    """Normalize varying HF label schemes to POSITIVE / NEGATIVE / NEUTRAL."""
    label = label.upper()
    if "POS" in label or label in ("LABEL_2", "LABEL_1") or label in ("4 STARS", "5 STARS"):
        return "POSITIVE", score
    if "NEG" in label or label == "LABEL_0" or label in ("1 STAR", "2 STARS"):
        return "NEGATIVE", score
    if "NEU" in label or "3 STARS" in label:
        return "NEUTRAL", score
    # fallback: star rating models return "1 star".."5 stars"
    if "STAR" in label:
        try:
            n = int(label[0])
            if n <= 2:
                return "NEGATIVE", score
            elif n == 3:
                return "NEUTRAL", score
            else:
                return "POSITIVE", score
        except Exception:
            pass
    return label, score


if run_button:
    with st.spinner(f"Loading {model_key} and running inference on {len(df_raw)} reviews..."):
        start = time.time()
        result_df = run_analysis(df_raw, model_key, batch_size)
        elapsed = time.time() - start
    st.session_state.results_df = result_df
    st.session_state.model_used = model_key
    st.success(f"✅ Done! Processed {len(result_df)} reviews in {elapsed:.1f}s using {model_key}.")

if compare_button:
    with st.spinner("Benchmarking DistilBERT, RoBERTa, and DeBERTa on a sample of 50 reviews..."):
        sample_df = df_raw.head(50).copy()
        bench_rows = []
        for mk in MODEL_OPTIONS.keys():
            start = time.time()
            out = run_analysis(sample_df, mk, batch_size=8)
            elapsed = time.time() - start
            bench_rows.append(
                {
                    "Model": mk,
                    "Avg Confidence": round(out["confidence"].mean(), 4),
                    "Positive %": round((out["sentiment"] == "POSITIVE").mean() * 100, 1),
                    "Negative %": round((out["sentiment"] == "NEGATIVE").mean() * 100, 1),
                    "Time (s) / 50 reviews": round(elapsed, 2),
                }
            )
        st.session_state.benchmark_df = pd.DataFrame(bench_rows)
    st.success("✅ Benchmark complete — see 'Model Comparison' tab.")

# --------------------------------------------------------------------------
# MAIN VIEWS
# --------------------------------------------------------------------------
# Show the active view based on sidebar selection
st.markdown(f"### {selected_tab}")

# ---------------- DASHBOARD VIEW ----------------
if selected_tab == "📈 Dashboard":
    if st.session_state.results_df is None:
        st.warning("Click **🚀 Run Sentiment Analysis** in the sidebar to process reviews and view the dashboard.")
    else:
        rdf = st.session_state.results_df

        col1, col2, col3, col4 = st.columns(4)
        total = len(rdf)
        pos_pct = (rdf["sentiment"] == "POSITIVE").mean() * 100
        neg_pct = (rdf["sentiment"] == "NEGATIVE").mean() * 100
        avg_conf = rdf["confidence"].mean() * 100

        col1.metric("Total Reviews", f"{total}")
        col2.metric("Positive %", f"{pos_pct:.1f}%")
        col3.metric("Negative %", f"{neg_pct:.1f}%")
        col4.metric("Avg. Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")

        c1, c2 = st.columns([1, 1])
        with c1:
            sentiment_counts = rdf["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Overall Sentiment Distribution",
                color="Sentiment",
                color_discrete_map={"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#f59e0b"},
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            cat_sentiment = rdf.groupby(["category", "sentiment"]).size().reset_index(name="count")
            fig_bar = px.bar(
                cat_sentiment,
                x="category",
                y="count",
                color="sentiment",
                title="Sentiment by Product Category",
                color_discrete_map={"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#f59e0b"},
                barmode="stack",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Trend over time
        rdf["date"] = pd.to_datetime(rdf["date"])
        trend = rdf.groupby([rdf["date"].dt.date, "sentiment"]).size().reset_index(name="count")
        trend.columns = ["date", "sentiment", "count"]
        fig_trend = px.line(
            trend,
            x="date",
            y="count",
            color="sentiment",
            title="Sentiment Trend Over Time",
            markers=True,
            color_discrete_map={"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#f59e0b"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # confidence distribution
        fig_hist = px.histogram(
            rdf, x="confidence", color="sentiment", nbins=30,
            title="Model Confidence Distribution",
            color_discrete_map={"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444", "NEUTRAL": "#f59e0b"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------- ASPECT ANALYSIS VIEW ----------------
elif selected_tab == "🏷️ Aspect Analysis":
    if st.session_state.results_df is None:
        st.warning("Click **🚀 Run Sentiment Analysis** in the sidebar first to unlock aspect-level insights.")
    else:
        rdf = st.session_state.results_df
        st.subheader("What are customers talking about?")

        aspects_pos = extract_aspects(rdf[rdf["sentiment"] == "POSITIVE"]["clean_review"].tolist())
        aspects_neg = extract_aspects(rdf[rdf["sentiment"] == "NEGATIVE"]["clean_review"].tolist())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👍 Top Praised Aspects")
            if aspects_pos:
                pos_df = pd.DataFrame(aspects_pos, columns=["Aspect", "Mentions"])
                fig = px.bar(pos_df.head(10).sort_values("Mentions"), x="Mentions", y="Aspect",
                             orientation="h", color_discrete_sequence=["#22c55e"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough positive reviews to extract aspects.")

        with c2:
            st.markdown("#### 👎 Top Complaint Aspects")
            if aspects_neg:
                neg_df = pd.DataFrame(aspects_neg, columns=["Aspect", "Mentions"])
                fig = px.bar(neg_df.head(10).sort_values("Mentions"), x="Mentions", y="Aspect",
                             orientation="h", color_discrete_sequence=["#ef4444"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough negative reviews to extract aspects.")

        st.markdown("---")
        st.markdown("#### 🔎 Aspect Explorer")
        all_aspects = sorted(set([a for a, _ in aspects_pos] + [a for a, _ in aspects_neg]))
        if all_aspects:
            chosen = st.selectbox("Pick an aspect to see related reviews", all_aspects)
            matched = rdf[rdf["clean_review"].str.contains(rf"\b{re.escape(chosen)}\b", case=False, na=False)]
            st.dataframe(matched[["review", "sentiment", "confidence"]], use_container_width=True)

# ---------------- BI REPORT VIEW ----------------
elif selected_tab == "📝 BI Report":
    if st.session_state.results_df is None:
        st.warning("Click **🚀 Run Sentiment Analysis** in the sidebar first to generate the business intelligence report.")
    else:
        rdf = st.session_state.results_df
        report_text = generate_bi_report(rdf, model_name=st.session_state.model_used)
        st.markdown(report_text)

        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=report_text,
            file_name="business_intelligence_report.md",
            mime="text/markdown",
        )

        csv_buf = io.StringIO()
        rdf.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download Full Results (CSV)",
            data=csv_buf.getvalue(),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )

# ---------------- MODEL COMPARISON VIEW ----------------
elif selected_tab == "⚖️ Model Comparison":
    st.subheader("Benchmark: RoBERTa vs DeBERTa vs DistilBERT")
    st.caption("Click 'Benchmark All 3 Models' in the sidebar to run this comparison on a 50-review sample.")
    if st.session_state.benchmark_df is not None:
        bdf = st.session_state.benchmark_df
        st.dataframe(bdf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(bdf, x="Model", y="Time (s) / 50 reviews", title="Inference Speed Comparison",
                         color="Model")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(bdf, x="Model", y="Avg Confidence", title="Average Confidence by Model",
                          color="Model")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No benchmark run yet. Click 'Benchmark All 3 Models' in the sidebar to compare performance.")

# ---------------- RAW DATA VIEW ----------------
elif selected_tab == "🗂️ Raw Data":
    st.subheader("🗂️ Dataset Overview")
    st.caption(f"Total reviews loaded: {len(df_raw)}")
    if st.session_state.results_df is not None:
        st.dataframe(st.session_state.results_df, use_container_width=True)
    else:
        st.dataframe(df_raw, use_container_width=True)

st.markdown("---")
st.caption("Mini Project: AI Agent for Customer Review Sentiment Analysis and Business Intelligence Generation | Models: RoBERTa · DeBERTa · DistilBERT | Dataset: Amazon Reviews (Kaggle)")
