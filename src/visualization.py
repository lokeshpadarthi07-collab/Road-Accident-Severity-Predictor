"""
Visualization Module for Road Accident Severity Predictor.

Produces publication-grade plots for Exploratory Data Analysis (EDA) and Model Evaluation.
Saves high-resolution figures to reports/figures/.
"""

import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set consistent, modern aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_PALETTE = ["#2b5c8f", "#d95f02", "#e7298a", "#7570b3", "#1b9e77", "#e6ab02"]


def ensure_output_dir(output_dir: str = "reports/figures") -> str:
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "accident_severity",
    output_dir: str = "reports/figures",
    save_filename: str = "01_class_distribution.png"
) -> str:
    """Plot target class distribution showing count and percentage, highlighting imbalance."""
    output_dir = ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, save_filename)

    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#3498db", "#e67e22", "#e74c3c"][:len(counts)], edgecolor="black", alpha=0.85)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_title("Accident Severity Target Distribution (Class Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Accident Severity Class", fontsize=11)
    ax.set_ylabel("Number of Incidents", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution plot to {save_path}")
    return save_path


def plot_accidents_by_hour(
    df: pd.DataFrame,
    hour_col: str = "hour_of_day",
    severity_col: str = "accident_severity",
    output_dir: str = "reports/figures",
    save_filename: str = "02_accidents_by_hour.png"
) -> str:
    """Plot accident frequency by hour of the day segmented by severity."""
    output_dir = ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, save_filename)

    fig, ax = plt.subplots(figsize=(10, 5))

    cross_tab = pd.crosstab(df[hour_col], df[severity_col])
    cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black", alpha=0.85)

    ax.set_title("Accident Frequency by Hour of Day & Severity", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (24-hour)", fontsize=11)
    ax.set_ylabel("Number of Incidents", fontsize=11)
    ax.legend(title="Severity", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved accidents by hour plot to {save_path}")
    return save_path


def plot_accidents_by_day(
    df: pd.DataFrame,
    day_col: str = "day_of_week",
    output_dir: str = "reports/figures",
    save_filename: str = "03_accidents_by_day.png"
) -> str:
    """Plot accident frequency across days of the week."""
    output_dir = ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, save_filename)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df[day_col].value_counts().reindex(day_order).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax, palette="Blues_r", edgecolor="black", alpha=0.85)

    for i, count in enumerate(day_counts.values):
        ax.text(i, count + 8, f"{int(count):,}", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Accident Frequency by Day of Week", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Total Recorded Incidents", fontsize=11)
    ax.set_ylim(0, max(day_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved accidents by day plot to {save_path}")
    return save_path


def plot_environmental_severity(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    output_dir: str = "reports/figures",
    save_filename: str = "04_environmental_severity.png"
) -> str:
    """Plot proportion of accident severities across environmental condition categories."""
    output_dir = ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, save_filename)

    # Calculate proportions per category
    prop_tab = pd.crosstab(df[category_col].fillna("Unknown"), df["accident_severity"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    prop_tab.plot(kind="barh", stacked=True, ax=ax, color=["#3498db", "#e67e22", "#e74c3c"], edgecolor="black", alpha=0.85)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Percentage of Incidents Within Category (%)", fontsize=11)
    ax.set_ylabel(category_col.replace("_", " ").title(), fontsize=11)
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved environmental severity plot to {save_path}")
    return save_path


def plot_speed_vs_severity(
    df: pd.DataFrame,
    output_dir: str = "reports/figures",
    save_filename: str = "05_speed_vs_severity.png"
) -> str:
    """Plot speed limit breakdown against severity."""
    output_dir = ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, save_filename)

    fig, ax = plt.subplots(figsize=(9, 5))
    cross = pd.crosstab(df["speed_limit"], df["accident_severity"], normalize="index") * 100
    cross.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22", "#e74c3c"], edgecolor="black", alpha=0.85)

    ax.set_title("Severity Proportions Across Speed Limits (mph)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Posted Speed Limit (mph)", fontsize=11)
    ax.set_ylabel("Proportion Within Speed Tier (%)", fontsize=11)
    ax.legend(title="Severity", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved speed vs severity plot to {save_path}")
    return save_path


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_dir: str = "reports/figures",
    save_filename: Optional[str] = None
) -> str:
    """Plot confusion matrix heatmap with raw counts and normalized percentages."""
    output_dir = ensure_output_dir(output_dir)
    if save_filename is None:
        save_filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, save_filename)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor="white"
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Severity", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Severity (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")
    return save_path


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    model_name: str,
    top_n: int = 15,
    output_dir: str = "reports/figures",
    save_filename: Optional[str] = None
) -> str:
    """Plot top N feature importances for a tree-based model."""
    output_dir = ensure_output_dir(output_dir)
    if save_filename is None:
        save_filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, save_filename)

    # Sort
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top_features))

    ax.barh(y_pos, top_vals[::-1], color="#2980b9", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("cat__", "").replace("num__", "").replace("remainder__", "") for f in top_features[::-1]])
    ax.set_xlabel("Relative Feature Importance (Gini / Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Influential Features: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")
    return save_path
