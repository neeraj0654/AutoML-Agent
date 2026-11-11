"""
Visualizations — generates matplotlib/seaborn charts and returns them as
base64-encoded PNG strings or matplotlib Figure objects for Streamlit.
"""
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = ["#6C63FF", "#FF6584", "#43BCCD", "#FFD166", "#06D6A0", "#EF476F"]
BG      = "#0E1117"
SURFACE = "#1A1D2E"
TEXT    = "#E0E0E0"
GRID    = "#2A2D3E"

def _style(fig, ax_or_axes=None):
    fig.patch.set_facecolor(BG)
    axes = [ax_or_axes] if ax_or_axes is not None and not hasattr(ax_or_axes, "__iter__") else (ax_or_axes or [])
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if hasattr(ax, "title"): ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Chart Builders ────────────────────────────────────────────────────────────

def missing_values_heatmap(df: pd.DataFrame):
    """Heatmap of missing value presence/absence."""
    missing_mask = df.isnull()
    if missing_mask.sum().sum() == 0:
        return None

    n_cols = min(len(df.columns), 30)
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.5), 4))
    sns.heatmap(
        missing_mask.iloc[:, :n_cols].T,
        ax=ax,
        cbar=False,
        cmap=sns.color_palette(["#1A1D2E", "#FF6584"], as_cmap=True),
        yticklabels=True,
        xticklabels=False,
    )
    ax.set_title("Missing Values Heatmap", color=TEXT, fontsize=12, pad=10)
    _style(fig, ax)
    return fig_to_buf(fig)


def distribution_grid(df: pd.DataFrame, numeric_cols: list, max_cols: int = 12):
    """Grid of histograms for numeric columns."""
    cols = numeric_cols[:max_cols]
    n = len(cols)
    if n == 0:
        return None
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="none")
        axes[i].set_title(col[:20], color=TEXT, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", color=TEXT, fontsize=13, y=1.01)
    _style(fig, list(axes))
    fig.tight_layout()
    return fig_to_buf(fig)


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Correlation matrix heatmap."""
    cols = numeric_cols[:20]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.6), max(5, len(cols) * 0.5)))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0, annot=len(cols) <= 12,
        fmt=".2f", linewidths=0.5, linecolor=BG,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7, "color": TEXT},
    )
    ax.set_title("Correlation Matrix", color=TEXT, fontsize=12, pad=10)
    _style(fig, ax)
    return fig_to_buf(fig)


def model_leaderboard_chart(leaderboard: list[dict], metric_name: str):
    """Horizontal bar chart of model scores."""
    names  = [r["model"] for r in leaderboard]
    scores = [r["cv_score"] for r in leaderboard]
    colors = [PALETTE[0] if i == 0 else PALETTE[2] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.55)))
    bars = ax.barh(names, scores, color=colors, edgecolor="none", height=0.5)
    ax.set_xlim(0, max(scores) * 1.15)
    ax.set_xlabel(metric_name, color=TEXT)
    ax.set_title("Model Leaderboard (CV Score)", color=TEXT, fontsize=12, pad=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", color=TEXT, fontsize=9)
    _style(fig, ax)
    fig.tight_layout()
    return fig_to_buf(fig)


def feature_importance_chart(importances: dict):
    """Bar chart for feature importances."""
    if not importances:
        return None
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    names  = [i[0] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    ax.set_xlabel("Importance", color=TEXT)
    ax.set_title("Top Feature Importances", color=TEXT, fontsize=12, pad=10)
    _style(fig, ax)
    fig.tight_layout()
    return fig_to_buf(fig)


def class_distribution(df: pd.DataFrame, target_col: str):
    """Bar chart for class distribution in classification tasks."""
    counts = df[target_col].value_counts()
    fig, ax = plt.subplots(figsize=(max(4, len(counts) * 0.8), 3.5))
    ax.bar(counts.index.astype(str), counts.values,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(counts))],
           edgecolor="none")
    ax.set_xlabel("Class", color=TEXT)
    ax.set_ylabel("Count", color=TEXT)
    ax.set_title(f"Target Distribution: {target_col}", color=TEXT, fontsize=12, pad=10)
    _style(fig, ax)
    fig.tight_layout()
    return fig_to_buf(fig)
