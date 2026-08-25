import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from llm_slr.config import FIGURES_DIR, N_SPLITS, RESULTS_DIR, THEMES
from llm_slr.data.loader import load_theme
from llm_slr.data.splits import YearsSplit
from llm_slr.eval.metrics import metrics_at

FIG_DIR = FIGURES_DIR

C_INCLUDED, C_EXCLUDED = "#2a78d6", "#c3c2b7"
C_SURFACE, C_INK, C_MUTED, C_GRID = "#ffffff", "#0b0b0b", "#898781", "#e1e0d9"
MODEL_COLORS = {"llama3.2:3b": "#2a78d6", "qwen2.5:3b": "#1baf7a",
                "gemma2:2b": "#eda100"}

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "text.color": C_INK, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
    "xtick.color": C_INK, "ytick.color": C_INK, "axes.axisbelow": True,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)


def fig_composicao(datasets):
    stats = {t: (sum(a.label for a in arts), len(arts))
             for t, arts in datasets.items()}
    order = sorted(stats, key=lambda t: stats[t][1])
    inc = [stats[t][0] for t in order]
    exc = [stats[t][1] - stats[t][0] for t in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(order, inc, color=C_INCLUDED, label="Incluídos na RSL",
            edgecolor=C_SURFACE, linewidth=2)
    ax.barh(order, exc, left=inc, color=C_EXCLUDED, label="Excluídos",
            edgecolor=C_SURFACE, linewidth=2)
    for t in order:
        pos, n = stats[t]
        ax.annotate(f"{pos} ({100 * pos / n:.0f}%)", (n, t), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("artigos candidatos")
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "composicao-datasets.png")


def fig_distribuicao_temporal(datasets):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, (theme, arts) in zip(axes.flat, datasets.items()):
        years = [a.year for a in arts]
        counts = pd.Series(years).value_counts().sort_index()
        visible = counts[counts.index >= 1990]
        ax.bar(visible.index, visible.values, color=C_INCLUDED, width=0.85)
        for _, test in YearsSplit(n_split=N_SPLITS, years=years).split(arts):
            ax.axvline(years[test[0]] - 0.5, color=C_MUTED,
                       linestyle="--", linewidth=1)
        hidden = int(counts[counts.index < 1990].sum())
        label = theme + (f"  (+{hidden} antes de 1990)" if hidden else "")
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    save(fig, "distribuicao-temporal.png")


def fig_dist_scores(raw):
    zs = raw[raw.strategy == "zero-shot"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    width = 0.38
    for ax, model in zip(axes, MODEL_COLORS):
        g = zs[zs.model == model]
        for shift, (label, color, name) in enumerate(
            [(1, C_INCLUDED, "Incluídos"), (0, C_EXCLUDED, "Excluídos")]
        ):
            sub = g[g.label == label]["score"].astype(int)
            frac = sub.value_counts(normalize=True).reindex(range(1, 8),
                                                            fill_value=0)
            xs = np.arange(1, 8) + (shift - 0.5) * width
            ax.bar(xs, frac.values, width=width, color=color, label=name,
                   edgecolor=C_SURFACE, linewidth=1)
        ax.set_title(model, loc="left", fontsize=10)
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("escore Likert")
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    axes[0].set_ylabel("fração dos artigos")
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "distribuicao-scores.png")


def fig_curva_operacao(raw):
    zs = raw[raw.strategy == "zero-shot"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for model, color in MODEL_COLORS.items():
        recalls, excludeds = [], []
        for t in range(1, 8):
            per_fold = []
            for (_, _), g in zs[zs.model == model].groupby(["theme", "fold"]):
                if g["label"].nunique() < 2:
                    continue
                per_fold.append(metrics_at(g["score"].astype(int).tolist(),
                                           g["label"].tolist(), t))
            recalls.append(np.mean([m.recall for m in per_fold]))
            excludeds.append(np.mean([(m.tn + m.fn) / m.n for m in per_fold]))
        axes[0].plot(range(1, 8), recalls, color=color, marker="o",
                     markersize=5, linewidth=2, label=model)
        axes[1].plot(range(1, 8), excludeds, color=color, marker="o",
                     markersize=5, linewidth=2, label=model)
    axes[0].axhline(0.95, color=C_MUTED, linestyle="--", linewidth=1)
    axes[0].annotate("recall mínimo 0,95", (4.4, 0.965), fontsize=8,
                     color=C_MUTED)
    axes[0].set_ylabel("recall médio")
    axes[1].set_ylabel("fração média não lida")
    for ax in axes:
        ax.set_xlabel("limiar de inclusão (escore $\\geq t$)")
        ax.set_xticks(range(1, 8))
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "curva-operacao.png")


def fig_boxplot_wss(summary):
    core = [a for a in summary["approach"].unique()
            if "-k1" not in a and "-k4" not in a]
    order = (summary[summary["approach"].isin(core)]
             .groupby("approach")["wss95"].mean().sort_values().index.tolist())
    data = [summary[summary["approach"] == a]["wss95"].values for a in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bp = ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
                    widths=0.55, medianprops={"color": C_INK})
    for patch, a in zip(bp["boxes"], order):
        model = a.split("/")[0]
        color = (MODEL_COLORS.get(model, C_EXCLUDED)
                 if "zero-shot" in a else "#dbe7f7"
                 if a != "svm-baseline" else C_EXCLUDED)
        patch.set_facecolor(color)
        patch.set_edgecolor(C_MUTED)
    for i, vals in enumerate(data, start=1):
        ax.plot(vals, np.full(len(vals), i), "o", color=C_INK, markersize=3,
                alpha=0.55)
    ax.axvline(0, color=C_MUTED, linewidth=0.8)
    ax.set_xlabel("WSS@95 (24 folds)")
    ax.grid(axis="x", color=C_GRID, linewidth=0.8)
    save(fig, "boxplot-wss95.png")


def _heatmap(pivot, fname, center):
    data = pivot.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list(
        "dv", ["#e34948", "#f0efec", "#1c5cab"])
    vmax = np.nanmax(np.abs(data - center)) or 1.0
    norm = TwoSlopeNorm(vcenter=center, vmin=center - vmax, vmax=center + vmax)
    fig, ax = plt.subplots(figsize=(9.5, 0.48 * len(pivot) + 1.6))
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d = abs(norm(data[i, j]) - 0.5)
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#ffffff" if d > 0.35 else C_INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, fname)


ORDER = ["llama3.2:3b/zero-shot", "qwen2.5:3b/zero-shot",
         "gemma2:2b/zero-shot",
         "llama3.2:3b/few-shot-random", "llama3.2:3b/few-shot-similar",
         "qwen2.5:3b/few-shot-random", "qwen2.5:3b/few-shot-similar",
         "gemma2:2b/few-shot-random", "gemma2:2b/few-shot-similar",
         "svm-baseline"]


def fig_heatmaps(summary):
    core = summary[summary["approach"].isin(ORDER)]
    wss = core.pivot_table(index="approach", columns="theme",
                           values="wss95", aggfunc="mean").reindex(ORDER)
    _heatmap(wss, "heatmap-wss95.png", center=0.0)
    auc = core.pivot_table(index="approach", columns="theme",
                           values="roc_auc", aggfunc="mean").reindex(ORDER)
    _heatmap(auc, "heatmap-auc.png", center=0.5)


def _variant(a):
    if a.endswith("zero-shot"):
        return ("zero", 0)
    for sel in ("random", "similar"):
        if sel in a:
            return (sel, 1 if "-k1" in a else 4 if "-k4" in a else 2)
    return (None, None)


def fig_dose_resposta(summary, metric, fname, ylabel):
    themes4 = ["slr", "illiterate", "testing", "xbi"]
    sub = summary[summary["theme"].isin(themes4)].copy()
    sub[["sel", "k"]] = pd.DataFrame([_variant(a) for a in sub["approach"]],
                                     index=sub.index)
    sub["model"] = sub["approach"].str.split("/").str[0]
    colors = {"random": "#2a78d6", "similar": "#1baf7a"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, model in zip(axes, ["llama3.2:3b", "qwen2.5:3b"]):
        m = sub[sub["model"] == model]
        zero = m[m["sel"] == "zero"][metric].mean()
        for sel in ("random", "similar"):
            pts = [(0, zero)] + [
                (k, m[(m["sel"] == sel) & (m["k"] == k)][metric].mean())
                for k in (1, 2, 4)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colors[sel], linewidth=2, marker="o",
                    markersize=6, label=f"few-shot {sel}")
        if metric == "roc_auc":
            ax.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=1)
        else:
            ax.axhline(0, color=C_GRID, linewidth=0.8)
        ax.set_xticks([0, 1, 2, 4], ["0\n(zero-shot)", "1", "2", "4"])
        ax.set_xlabel("exemplos por classe (k)")
        ax.set_title(model, loc="left", fontsize=10)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, fname)


def main():
    global FIG_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.environ.get("LLM_SLR_FIG_DIR"),
                        help="diretório de saída "
                             f"(padrão: {FIGURES_DIR}, ou $LLM_SLR_FIG_DIR)")
    args = parser.parse_args()
    if args.out:
        FIG_DIR = Path(args.out)

    datasets = {t: load_theme(t) for t in sorted(THEMES)}
    raw = pd.read_csv(RESULTS_DIR / "raw.csv")
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")

    print(f"gravando figuras em {FIG_DIR}", flush=True)
    fig_composicao(datasets)
    fig_distribuicao_temporal(datasets)
    fig_dist_scores(raw)
    fig_curva_operacao(raw)
    fig_boxplot_wss(summary)
    fig_heatmaps(summary)
    fig_dose_resposta(summary, "wss95", "dose-resposta-k.png",
                      "WSS@95 médio")
    fig_dose_resposta(summary, "roc_auc", "dose-resposta-auc.png",
                      "ROC-AUC médio")


if __name__ == "__main__":
    main()
