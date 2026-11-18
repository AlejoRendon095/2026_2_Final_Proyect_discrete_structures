"""
Comparative analysis of all models (T8, Juan Diego Balanta).

Every model notebook saves ``outputs/metrics/<model>_metrics.json`` with, at least::

    {
      "model": "LSTM",                      # display name
      "train": {"accuracy": .., "precision": .., "recall": .., "f1": .., "kappa": ..},
      "test":  {"accuracy": .., "precision": .., "recall": .., "f1": .., "kappa": ..},
      "train_time_sec": 12.3,               # optional, needed for the cost plot
      "parameters": 123456                  # optional, int or {"total": int, ...}
    }

``train`` / ``test`` are the dicts returned by ``src.evaluate.evaluate``. Files
without a ``test`` key (e.g. grid-search logs) are ignored.

Functions
---------
load_all_metrics          read every *_metrics.json into a list of dicts
comparison_table          tidy DataFrame: one row per model, train/test metrics
to_markdown / to_latex    table ready for the README or the report
plot_test_metrics         grouped bars of the 5 test metrics per model
plot_generalization_gap   train vs test F1 (overfitting view)
plot_cost_vs_performance  parameters (log) vs test F1, marker size = training time
make_all_figures          everything above in one call
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["accuracy", "precision", "recall", "f1", "kappa"]

# Fixed order and colors so every figure in the report reads the same way.
MODEL_ORDER = ["Dummy (most_frequent)", "Dummy (stratified)", "Dense NN", "RNN", "LSTM", "DistilBERT"]
PALETTE = {
    "Dummy (stratified)": "#cfcfcf",
    "Dummy": "#8a8a8a",
    "Dense": "#4c72b0",
    "RNN": "#dd8452",
    "LSTM": "#55a868",
    "DistilBERT": "#8172b3",
}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _short_name(name: str) -> str:
    """Normalizes the names each teammate may use ("lstm", "LSTM model", ...)."""
    low = name.lower()
    if "dummy" in low or "baseline" in low:
        if "strat" in low:
            return "Dummy (stratified)"
        return "Dummy (most_frequent)"
    if "bert" in low or "transformer" in low:
        return "DistilBERT"
    if "lstm" in low:
        return "LSTM"
    if "rnn" in low:
        return "RNN"
    if "dense" in low or "mlp" in low or "feedforward" in low:
        return "Dense NN"
    return name


def _color(model: str) -> str:
    for key, col in PALETTE.items():
        if key.lower() in model.lower():
            return col
    return "#333333"


def _n_params(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("total") or value.get("trainable")
    return int(value)


def load_all_metrics(metrics_dir: PathLike = "outputs/metrics") -> List[Dict]:
    """Reads every ``*_metrics.json`` that has a ``test`` section."""
    results = []
    for path in sorted(Path(metrics_dir).glob("*_metrics.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # a file may hold a single model or a list of models (e.g. two dummies)
        for item in data if isinstance(data, list) else [data]:
            if "test" not in item:
                continue
            item = dict(item)
            item["model"] = _short_name(item.get("model", path.stem.replace("_metrics", "")))
            item["_source"] = path.name
            results.append(item)
    return results


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def comparison_table(results: Iterable[Dict]) -> pd.DataFrame:
    """One row per model with train_/test_ metrics, F1 gap, time and parameters."""
    rows = []
    for r in results:
        row = {"model": r["model"]}
        for split in ("train", "test"):
            for m in METRICS:
                row[f"{split}_{m}"] = r.get(split, {}).get(m, np.nan)
        row["f1_gap"] = row["train_f1"] - row["test_f1"]
        row["train_time_sec"] = r.get("train_time_sec", np.nan)
        row["parameters"] = _n_params(r.get("parameters"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_o"] = df["model"].map(lambda m: order.get(m, len(order)))
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def to_markdown(df: pd.DataFrame, split: str = "test", decimals: int = 3) -> str:
    """Compact table of one split, best value per column in bold."""
    cols = [f"{split}_{m}" for m in METRICS]
    out = df[["model"] + cols].copy()
    header = "| Model | " + " | ".join(m.capitalize() if m != "f1" else "F1" for m in METRICS) + " |"
    lines = [header, "|---" * (len(METRICS) + 1) + "|"]
    best = {c: out[c].max() for c in cols}
    for _, row in out.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            s = "–" if pd.isna(v) else f"{v:.{decimals}f}"
            cells.append(f"**{s}**" if not pd.isna(v) and np.isclose(v, best[c]) else s)
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, decimals: int = 3) -> str:
    """Train/test table for the report (booktabs)."""
    cols = ["model"] + [f"test_{m}" for m in METRICS] + ["train_f1", "f1_gap"]
    t = df[cols].copy()
    t.columns = ["Model", "Acc.", "Prec.", "Rec.", "F1", "$\\kappa$", "Train F1", "F1 gap"]
    return t.to_latex(index=False, float_format=f"%.{decimals}f", na_rep="--", escape=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_test_metrics(df: pd.DataFrame, ax=None):
    """Grouped bars: x = metric, one bar per model."""
    ax = ax or plt.subplots(figsize=(8, 3.6))[1]
    n = len(df)
    width = 0.8 / max(n, 1)
    x = np.arange(len(METRICS))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[f"test_{m}"] for m in METRICS]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=row["model"], color=_color(row["model"]))
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "Kappa"])
    ax.set_ylim(min(0, np.nanmin(df[[f"test_{m}" for m in METRICS]].values)) - 0.02, 1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("score (test)")
    ax.set_title("Test performance by model")
    ax.legend(fontsize=8, ncol=min(n, 6), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_generalization_gap(df: pd.DataFrame, ax=None):
    """Train F1 vs test F1 per model; a long segment means overfitting."""
    ax = ax or plt.subplots(figsize=(6, 3.4))[1]
    y = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        c = _color(row["model"])
        ax.plot([row["test_f1"], row["train_f1"]], [i, i], color=c, lw=2, alpha=0.6)
        ax.scatter(row["train_f1"], i, color=c, marker="o", s=40, zorder=3)
        ax.scatter(row["test_f1"], i, color=c, marker="D", s=40, zorder=3, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_title("Generalization: train (o) vs test (◆) F1")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_cost_vs_performance(df: pd.DataFrame, ax=None):
    """Number of parameters (log scale) vs test F1; marker area ~ training time."""
    ax = ax or plt.subplots(figsize=(6, 3.6))[1]
    d = df.dropna(subset=["parameters"])
    d = d[d["parameters"] > 0]
    if d.empty:
        ax.text(0.5, 0.5, "no parameter counts yet", ha="center", va="center", transform=ax.transAxes)
        return ax
    t = d["train_time_sec"].fillna(d["train_time_sec"].median() if d["train_time_sec"].notna().any() else 1)
    sizes = 40 + 360 * (np.log1p(t) / max(np.log1p(t).max(), 1e-9))
    d = d.assign(_size=sizes)
    for _, row in d.iterrows():
        s = row["_size"]
        ax.scatter(row["parameters"], row["test_f1"], s=s, color=_color(row["model"]), alpha=0.8,
                   edgecolor="black")
        label = row["model"]
        if not pd.isna(row["train_time_sec"]):
            label += f" · {row['train_time_sec']:.0f}s"
        # Label on the left when a slightly bigger model sits next to it (its label
        # would otherwise land on that neighbour's marker); otherwise on the right.
        p = row["parameters"]
        crowded = ((d["parameters"] > p) & (d["parameters"] <= 2.5 * p)).any()
        ax.annotate(label, (p, row["test_f1"]), textcoords="offset points",
                    xytext=(-12, -3) if crowded else (12, -3), ha="right" if crowded else "left",
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(d["parameters"].min() / 10, d["parameters"].max() * 6)
    pad = max(0.02, (d["test_f1"].max() - d["test_f1"].min()) * 0.15)
    ax.set_ylim(d["test_f1"].min() - pad, d["test_f1"].max() + pad)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("test F1")
    ax.set_title("Cost vs performance (marker size = training time)")
    ax.grid(alpha=0.3, which="both")
    return ax


def make_all_figures(metrics_dir: PathLike = "outputs/metrics",
                     fig_dir: PathLike = "outputs/figures") -> pd.DataFrame:
    """Builds the table, saves the CSV/markdown/LaTeX versions and the three figures."""
    metrics_dir, fig_dir = Path(metrics_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = comparison_table(load_all_metrics(metrics_dir))
    if df.empty:
        print(f"no *_metrics.json with a 'test' section in {metrics_dir}")
        return df

    df.to_csv(metrics_dir / "comparison_table.csv", index=False)
    (metrics_dir / "comparison_table.md").write_text(to_markdown(df), encoding="utf-8")
    (metrics_dir / "comparison_table.tex").write_text(to_latex(df), encoding="utf-8")

    for name, fn, size in [("comparison_test_metrics", plot_test_metrics, (8, 4)),
                           ("comparison_generalization_gap", plot_generalization_gap, (6, 3.4)),
                           ("comparison_cost_vs_performance", plot_cost_vs_performance, (6, 3.6))]:
        fig, ax = plt.subplots(figsize=size)
        fn(df, ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=200)
        plt.close(fig)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the comparison table and figures.")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--fig-dir", default="outputs/figures")
    a = p.parse_args()
    table = make_all_figures(a.metrics_dir, a.fig_dir)
    if not table.empty:
        print(to_markdown(table))
