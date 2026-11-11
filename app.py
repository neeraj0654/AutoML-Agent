"""
AutoML Agent for Tabular Data — Streamlit UI
"""
import io
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0E1117 0%, #12152B 50%, #0E1117 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12152B 0%, #1A1D2E 100%);
    border-right: 1px solid #2A2D4E;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #6C63FF;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF22 0%, #FF658422 50%, #43BCCD22 100%);
    border: 1px solid #6C63FF44;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6C63FF, #FF6584, #43BCCD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: #9090B0;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1A1D2E, #22253A);
    border: 1px solid #2A2D4E;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #6C63FF;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #6C63FF;
}
.metric-label {
    font-size: 0.8rem;
    color: #9090B0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Best model badge */
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    margin: 0.3rem 0;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #6C63FF;
    border-bottom: 2px solid #6C63FF44;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Info boxes */
.info-box {
    background: #1A1D2E;
    border-left: 4px solid #6C63FF;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    color: #C0C0D8;
    font-size: 0.9rem;
}

/* Step pill */
.step-pill {
    display: inline-block;
    background: #6C63FF22;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    color: #A09FF0;
    margin: 0.2rem 0.1rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #6C63FF55 !important;
    border-radius: 12px !important;
    background: #1A1D2E !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #12152B;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #9090B0;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #8B84FF) !important;
    color: white !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6C63FF, #FF6584) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #12152B; }
::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Imports (lazy to keep startup fast) ──────────────────────────────────────
from agents.orchestrator import run_pipeline
from agents import data_profiler
from utils import visualizations as viz
from utils.pdf_export import build_pdf
from utils.llm_client import is_available as llm_available


# ── Session state init ────────────────────────────────────────────────────────
for key in ["results", "df", "dataset_name"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 AutoML Agent")
    st.markdown("---")

    # ── API Key ───────────────────────────────────────────────────────────────
    with st.expander("🔑 Gemini API Key", expanded=not llm_available()):
        api_key_input = st.text_input(
            "API Key", type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="AIzaSy...",
            help="Get a free key at https://aistudio.google.com/",
        )
        if api_key_input and api_key_input != os.getenv("GEMINI_API_KEY", ""):
            os.environ["GEMINI_API_KEY"] = api_key_input
            import utils.llm_client as _lc
            _lc._init()
            st.success("API key updated!")

    llm_status = "✅ LLM Connected" if llm_available() else "⚠️ No API Key (using fallback)"
    st.markdown(f'<div class="info-box">{llm_status}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        help="Upload any tabular CSV dataset.",
    )

    # ── Sample datasets ───────────────────────────────────────────────────────
    st.markdown("#### Or use a sample dataset")
    sample_choice = st.selectbox(
        "Sample",
        ["— None —", "Iris (Classification)", "California Housing (Regression)"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    test_size  = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    cv_folds   = st.slider("CV folds", 3, 10, 5)

    target_col_manual = st.text_input(
        "Target column (optional)",
        placeholder="Leave blank for auto-detect",
        help="Override the auto-detected target column.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run AutoML Pipeline", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
df = None
dataset_name = "Dataset"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name.replace(".csv", "")
elif sample_choice == "Iris (Classification)":
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "iris.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        dataset_name = "Iris"
    else:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = [*iris.feature_names, "species"]
        dataset_name = "Iris"
elif sample_choice == "California Housing (Regression)":
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "california_housing.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        dataset_name = "California Housing"
    else:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        dataset_name = "California Housing"


# ════════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🤖 AutoML Agent</p>
    <p class="hero-sub">Upload a CSV → Profile → Engineer Features → Train Models → Get AI Report</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# DATA PREVIEW (before running pipeline)
# ════════════════════════════════════════════════════════════════════════════════
if df is not None and st.session_state.results is None:
    st.markdown(f"### 📂 `{dataset_name}` — {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True, height=280)


# ════════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
if run_btn:
    if df is None:
        st.error("⚠️ Please upload a CSV file or select a sample dataset first.")
    else:
        st.session_state.df = df
        st.session_state.dataset_name = dataset_name

        status_box = st.empty()
        progress_bar = st.progress(0)
        log_box = st.empty()

        def progress_callback(stage, pct, msg):
            progress_bar.progress(pct / 100)
            status_box.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

        target = target_col_manual.strip() or None
        results = run_pipeline(
            df=df,
            dataset_name=dataset_name,
            target_col=target,
            cv_folds=cv_folds,
            test_size=test_size,
            progress_callback=progress_callback,
        )
        st.session_state.results = results
        progress_bar.empty()
        status_box.empty()

        if results.get("error"):
            st.error(f"Pipeline failed:\n{results['error']}")
        else:
            st.success("✅ Pipeline complete! Scroll down to explore results.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ════════════════════════════════════════════════════════════════════════════════
results = st.session_state.results
if results and not results.get("error"):
    profile        = results["profile"]
    feature_plan   = results["feature_plan"]
    trainer_result = results["trainer_result"]
    importance_dict = results["importance_dict"]
    report_text    = results["report_text"]
    ds_name        = results["dataset_name"]
    df_used        = st.session_state.df

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, str(profile["n_rows"]),         "Rows"),
        (c2, str(profile["n_cols"]),          "Columns"),
        (c3, str(profile["total_missing"]),   "Missing Cells"),
        (c4, profile["task_type"].title(),    "Task Type"),
        (c5, f"{trainer_result['best_score']:.4f}", trainer_result["metric_name"]),
    ]
    for col, val, label in cards:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Data Profile",
        "🛠  Feature Engineering",
        "🏆  Model Leaderboard",
        "📄  AI Report",
    ])

    # ── TAB 1: Data Profile ───────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
        st.markdown(data_profiler.summary_text(profile))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Column Types</div>', unsafe_allow_html=True)
            type_df = pd.DataFrame({
                "Column": df_used.columns,
                "dtype":  df_used.dtypes.astype(str).values,
                "Missing": [profile["missing_per_col"].get(c, 0) for c in df_used.columns],
                "Missing %": [profile["missing_pct"].get(c, 0.0) for c in df_used.columns],
            })
            st.dataframe(type_df, use_container_width=True, height=320)

        with col_b:
            st.markdown('<div class="section-header">Numeric Statistics</div>', unsafe_allow_html=True)
            if profile["numeric_stats"]:
                stats_df = pd.DataFrame(profile["numeric_stats"]).T.round(3)
                st.dataframe(stats_df, use_container_width=True, height=320)
            else:
                st.info("No numeric columns found.")

        # Missing values heatmap
        st.markdown('<div class="section-header">Missing Values Heatmap</div>', unsafe_allow_html=True)
        mv_buf = viz.missing_values_heatmap(df_used)
        if mv_buf:
            st.image(mv_buf, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")

        # Distributions
        st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
        dist_buf = viz.distribution_grid(df_used, profile["numeric_cols"])
        if dist_buf:
            st.image(dist_buf, use_container_width=True)

        # Correlation
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        corr_buf = viz.correlation_heatmap(df_used, profile["numeric_cols"])
        if corr_buf:
            st.image(corr_buf, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation matrix.")

        # Class distribution (classification only)
        if profile["task_type"] == "classification" and profile.get("target_col"):
            st.markdown('<div class="section-header">Target Class Distribution</div>', unsafe_allow_html=True)
            cd_buf = viz.class_distribution(df_used, profile["target_col"])
            if cd_buf:
                st.image(cd_buf, use_container_width=True)

    # ── TAB 2: Feature Engineering ────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Pipeline Steps Applied</div>', unsafe_allow_html=True)
        for step in feature_plan.get("steps_applied", []):
            st.markdown(f'<span class="step-pill">✓ {step}</span>', unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown('<div class="section-header">🤖 AI Feature Engineering Suggestions</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-box">{feature_plan.get("llm_suggestions", "No suggestions available.")}</div>',
            unsafe_allow_html=True,
        )

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown('<div class="section-header">Numeric Features</div>', unsafe_allow_html=True)
            st.write(feature_plan.get("numeric_features", []))
        with col_y:
            st.markdown('<div class="section-header">Categorical Features</div>', unsafe_allow_html=True)
            st.write(feature_plan.get("categorical_features", []))

        st.markdown('<div class="section-header">Engineered Feature Names (after encoding)</div>', unsafe_allow_html=True)
        feat_names = feature_plan.get("feature_names", [])
        st.caption(f"{len(feat_names)} total features after preprocessing")
        st.code(", ".join(feat_names[:50]) + ("..." if len(feat_names) > 50 else ""), language=None)

    # ── TAB 3: Model Leaderboard ──────────────────────────────────────────────
    with tab3:
        leaderboard = trainer_result["leaderboard"]
        metric_name = trainer_result["metric_name"]
        best_name   = trainer_result["best_model_name"]
        best_score  = trainer_result["best_score"]

        st.markdown(
            f'<div style="text-align:center; margin-bottom:1rem;">'
            f'<span style="color:#9090B0; font-size:0.9rem;">Best Model</span><br>'
            f'<span class="best-badge">🥇 {best_name}  —  {metric_name}: {best_score:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Leaderboard table
        st.markdown('<div class="section-header">All Models Ranked</div>', unsafe_allow_html=True)
        lb_df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Model": r["model"],
                f"CV {metric_name}": round(r["cv_score"], 4),
                "CV Std": round(r["cv_std"], 4),
                "Fit Time (s)": round(r.get("fit_time", 0), 2),
            }
            for i, r in enumerate(leaderboard)
        ])
        st.dataframe(
            lb_df.style.highlight_max(subset=[f"CV {metric_name}"], color="#6C63FF44"),
            use_container_width=True, hide_index=True,
        )

        # Leaderboard chart
        lb_buf = viz.model_leaderboard_chart(leaderboard, metric_name)
        if lb_buf:
            st.image(lb_buf, use_container_width=True)

        # Test-set metrics
        st.markdown('<div class="section-header">Test Set Performance (Best Model)</div>', unsafe_allow_html=True)
        test_metrics = trainer_result.get("test_metrics", {})
        if profile["task_type"] == "regression":
            m1, m2, m3 = st.columns(3)
            m1.metric("R²",   f"{test_metrics.get('r2',  0):.4f}")
            m2.metric("MAE",  f"{test_metrics.get('mae', 0):.4f}")
            m3.metric("RMSE", f"{test_metrics.get('rmse',0):.4f}")
        else:
            cr = test_metrics.get("classification_report", {})
            wa = cr.get("weighted avg", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Weighted F1",        f"{wa.get('f1-score',  0):.4f}")
            m2.metric("Weighted Precision",  f"{wa.get('precision', 0):.4f}")
            m3.metric("Weighted Recall",     f"{wa.get('recall',    0):.4f}")

        # Feature importances
        if importance_dict:
            st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
            imp_buf = viz.feature_importance_chart(importance_dict)
            if imp_buf:
                st.image(imp_buf, use_container_width=True)

    # ── TAB 4: AI Report ──────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🤖 AI-Generated Analysis</div>', unsafe_allow_html=True)
        if not llm_available():
            st.info("💡 Add a Gemini API key in the sidebar for a full AI-generated narrative.")
        st.markdown(report_text or "_No report generated._")

        st.markdown("---")
        st.markdown("### 📥 Download Report as PDF")

        # Build charts for PDF
        pdf_charts = {}
        lb_buf2 = viz.model_leaderboard_chart(trainer_result["leaderboard"], trainer_result["metric_name"])
        if lb_buf2:
            pdf_charts["leaderboard"] = lb_buf2
        if importance_dict:
            imp_buf2 = viz.feature_importance_chart(importance_dict)
            if imp_buf2:
                pdf_charts["importance"] = imp_buf2

        try:
            pdf_buf = build_pdf(
                dataset_name=ds_name,
                profile=profile,
                feature_plan=feature_plan,
                leaderboard=trainer_result["leaderboard"],
                best_model_name=trainer_result["best_model_name"],
                best_score=trainer_result["best_score"],
                metric_name=trainer_result["metric_name"],
                report_text=report_text or "",
                chart_bufs=pdf_charts,
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buf,
                file_name=f"automl_report_{ds_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    # ── Reset ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Reset & Upload New Dataset", use_container_width=True):
        st.session_state.results = None
        st.session_state.df = None
        st.session_state.dataset_name = None
        st.rerun()

elif df is None and st.session_state.results is None:
    # ── Welcome screen ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #9090B0;">
        <div style="font-size:4rem; margin-bottom:1rem;">📂</div>
        <h3 style="color: #C0C0D8;">Upload a CSV to get started</h3>
        <p>Or select a sample dataset from the sidebar</p>
    </div>

    <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:1rem; margin-top:2rem;">
    """, unsafe_allow_html=True)

    features = [
        ("🔍", "Data Profiling", "Missing values, distributions, correlations"),
        ("🛠", "Feature Engineering", "Auto imputation, encoding, scaling"),
        ("🏆", "Model Training", "6+ models, 5-fold CV, XGBoost + sklearn"),
        ("📄", "AI Reports", "LLM-written narratives + PDF export"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2rem;">{icon}</div>'
            f'<div style="color:#C0C0D8; font-weight:600; margin:0.5rem 0;">{title}</div>'
            f'<div style="color:#707090; font-size:0.82rem;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
