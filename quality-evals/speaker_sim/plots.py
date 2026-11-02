"""Plots for analyze.py. One PNG per question; every value is also in the CSVs."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Fixed slot order: "all" is always ink, categories always take the same hue.
CAT_COLORS = {"all": INK, "short": "#2a78d6", "medium": "#eb6834", "long": "#1baf7a"}
# Other category names (e.g. content types) take the fixed categorical order, in sorted order.
SPARE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"sim_ref": "SIM vs reference voice", "sim_hi_nfe": "SIM vs same prompt at highest NFE",
          "wer": "WER", "latency_ms": "Latency (ms)", "rtf": "Real-time factor",
          "failed": "Share of failed (silent) clips", "cer": "CER",
          "wer_heard": "WER on words spoken before cut-off", "truncated": "Share of clips cut off before the end"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _color(cat, cats):
    if cat in CAT_COLORS:
        return CAT_COLORS[cat]
    others = [c for c in cats if c not in CAT_COLORS]
    return SPARE[others.index(cat) % len(SPARE)]


def _rows(summary, run, metric, cat=None):
    r = [s for s in summary if s["run_id"] == run and s["metric"] == metric and (cat is None or s["category"] == cat)]
    return sorted(r, key=lambda s: s["nfe"])


def _finish(fig, title, run, note="mean ± 95% CI"):
    """Title + subtitle in a fixed-height header band, then lay out the plots below it."""
    lines = [f"run: {run} · {note}"]
    if run.startswith("dummy"):
        lines.append("SYNTHETIC TEST DATA, not real results")
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.12 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.015, 1 - 0.42 / h, "\n".join(lines), ha="left", va="top", fontsize=9, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.6 + 0.18 * len(lines)) / h))


def _nfe_axis(ax, nfes):
    if min(nfes) > 0 and max(nfes) / min(nfes) >= 8:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("NFE (log scale)")
    else:
        ax.set_xlabel("NFE")
    ax.set_xticks(nfes, [str(n) for n in nfes])
    ax.minorticks_off()


def _series(ax, summary, run, metric, legend=True):
    cats = sorted({s["category"] for s in summary if s["run_id"] == run}, key=lambda c: (c != "all", c))
    order = {"all": 0, "short": 1, "medium": 2, "long": 3}
    cats.sort(key=lambda c: order.get(c, 9))
    for cat in cats:
        r = _rows(summary, run, metric, cat)
        if not r:
            continue
        x = [s["nfe"] for s in r]
        col = _color(cat, cats)
        ax.fill_between(x, [s["ci_low"] for s in r], [s["ci_high"] for s in r], color=col, alpha=0.10, lw=0)
        ax.plot(x, [s["mean"] for s in r], color=col, lw=2.5 if cat == "all" else 2,
                marker="o", ms=8, mec=SURFACE, mew=2, label="all prompts" if cat == "all" else cat, zorder=3)
    if legend and len(cats) > 1:
        ax.legend(loc="best")


def nfe_vs(summary, run, metrics, path, title, note="mean ± 95% CI"):
    metrics = [m for m in metrics if _rows(summary, run, m)]
    if not metrics:
        return None
    nfes = sorted({s["nfe"] for s in summary if s["run_id"] == run})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        _series(ax, summary, run, m, legend=False)
        _nfe_axis(ax, nfes)
        ax.set_ylabel(LABELS[m])
    handles, names = axes[0][0].get_legend_handles_labels()
    if len(handles) > 1:
        fig.legend(handles, names, loc="upper right", ncol=len(handles), bbox_to_anchor=(0.995, 1 - 0.3 / fig.get_figheight()))
    _finish(fig, title, run, note)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def nfe_vs_latency(summary, fits, run, path):
    if not _rows(summary, run, "latency_ms"):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _series(ax, summary, run, "latency_ms")
    nfes = sorted({s["nfe"] for s in summary if s["run_id"] == run})
    ax.set_xlabel("NFE")
    ax.set_xticks(nfes, [str(n) for n in nfes])
    ax.set_ylabel(LABELS["latency_ms"])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fit = next((f for f in fits if f["run_id"] == run and f["category"] == "all" and f["metric"] == "latency_ms"), None)
    if fit:
        xs = np.array([0, max(nfes)])
        ax.plot(xs, fit["intercept"] + fit["per_nfe"] * xs, color=MUTED, lw=1, zorder=2)
        ax.text(0.02, 0.97, f"fit (all prompts): {fit['intercept']:.0f} ms fixed overhead\n"
                            f"+ {fit['per_nfe']:.1f} ms per NFE  (R² {fit['r2']:.2f})",
                transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    _finish(fig, "Latency vs NFE", run, "mean ± 95% CI · grey line = linear fit; its value at NFE 0 is fixed overhead")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def latency_vs_quality(summary, run, path):
    lat = {s["nfe"]: s for s in _rows(summary, run, "latency_ms", "all")}
    quality = [m for m in ("sim_ref", "sim_hi_nfe", "wer") if _rows(summary, run, m, "all")]
    if not lat or not quality:
        return None
    if "sim_ref" in quality and "sim_hi_nfe" in quality:
        quality.remove("sim_hi_nfe")  # one SIM panel is enough here
    fig, axes = plt.subplots(1, len(quality), figsize=(max(5.2 * len(quality), 7.5), 4.2), squeeze=False)
    for ax, m in zip(axes[0], quality):
        pts = [(lat[s["nfe"]], s) for s in _rows(summary, run, m, "all") if s["nfe"] in lat]
        x = np.array([l["mean"] for l, _ in pts])
        y = np.array([q["mean"] for _, q in pts])
        xerr = np.array([[l["mean"] - l["ci_low"], l["ci_high"] - l["mean"]] for l, _ in pts]).T
        yerr = np.array([[q["mean"] - q["ci_low"], q["ci_high"] - q["mean"]] for _, q in pts]).T
        ax.errorbar(x, y, xerr=np.nan_to_num(xerr), yerr=np.nan_to_num(yerr), color=CAT_COLORS["short"],
                    lw=2, elinewidth=1, ecolor=AXIS, marker="o", ms=8, mec=SURFACE, mew=2, capsize=0, zorder=3)
        # Label selectively: ends always, others only if far enough from the last label to stay readable.
        span, last = (x.max() - x.min()) or 1, None
        for k, ((l, q), xi, yi) in enumerate(zip(pts, x, y)):
            if k in (0, len(pts) - 1) or last is None or (xi - last) / span > 0.14:
                if k == len(pts) - 2 and (x[-1] - xi) / span < 0.14:
                    continue
                ax.annotate(f"NFE {q['nfe']}", (xi, yi), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK2)
                last = xi
        ax.set_xlabel(LABELS["latency_ms"])
        ax.set_ylabel(LABELS[m] + (" (lower is better)" if m == "wer" else " (higher is better)"))
    _finish(fig, "Latency vs quality: where does extra compute stop paying off?", run,
            "all prompts, mean ± 95% CI · look for the knee")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def prompt_heatmap(clips, run, path, metric="sim_hi_nfe"):
    rows = [c for c in clips if c["run_id"] == run and not math.isnan(c[metric])]
    nfes = sorted({c["nfe"] for c in rows})
    prompts = sorted({c["prompt_id"] for c in rows})
    if len(nfes) < 2 or not prompts:
        return None
    grid = np.full((len(prompts), len(nfes)), np.nan)
    for i, p in enumerate(prompts):
        for j, n in enumerate(nfes):
            v = [c[metric] for c in rows if c["prompt_id"] == p and c["nfe"] == n]
            if v:
                grid[i, j] = np.mean(v)  # averaged over seeds
    order = np.argsort(grid[:, 0])  # worst at the lowest NFE first
    grid, prompts = grid[order], [prompts[i] for i in order]
    cat = {c["prompt_id"]: c["category"] for c in rows}
    fig, ax = plt.subplots(figsize=(1.1 * len(nfes) + 3.4, 0.24 * len(prompts) + 1.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(len(nfes)), [str(n) for n in nfes])
    ax.set_yticks(range(len(prompts)), [f"{p} ({cat[p]})" if cat[p] != "all" else p for p in prompts], fontsize=8)
    ax.set_xlabel("NFE")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(LABELS[metric], color=INK2)
    cb.outline.set_visible(False)
    _finish(fig, "Which prompts degrade first?", run, f"mean over seeds · sorted by score at NFE {nfes[0]}, worst first")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def compare_runs(summary, runs, out_dir) -> list[Path]:
    """Overall mean per NFE for each run, one panel per metric (same axis per metric, never two)."""
    metrics = [m for m in ("sim_ref", "wer", "truncated", "rtf", "failed") if any(_rows(summary, r, m, "all") for r in runs)]
    for m in ("truncated", "failed"):  # only worth a panel if it ever happens
        if m in metrics and not any(s["mean"] > 0 for r in runs for s in _rows(summary, r, m, "all")):
            metrics.remove(m)
    if not metrics:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nfes = sorted({s["nfe"] for s in summary if s["run_id"] in runs})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for i, run in enumerate(runs):
            r = _rows(summary, run, m, "all")
            x = [s["nfe"] for s in r]
            col = SPARE[i % len(SPARE)]
            ax.fill_between(x, [s["ci_low"] for s in r], [s["ci_high"] for s in r], color=col, alpha=0.10, lw=0)
            ax.plot(x, [s["mean"] for s in r], color=col, marker="o", ms=8, mec=SURFACE, mew=2, label=run, zorder=3)
        if m == "rtf":  # cost is linear in NFE, so a linear axis shows it as a straight line
            ax.set_xticks(nfes, [str(n) for n in nfes])
            ax.set_xlabel("NFE")
            ax.axhline(1, color=MUTED, lw=1, zorder=2)
            ax.text(nfes[-1], 1.08, "real time (RTF 1) ", ha="right", va="bottom", fontsize=8.5, color=INK2)
            ax.set_ylim(bottom=0)
        else:
            _nfe_axis(ax, nfes)
        if m in ("failed", "truncated", "wer"):
            ax.set_ylim(bottom=0)
        ax.set_ylabel(LABELS[m])
    axes[0][0].legend(loc="best")
    _finish(fig, "Runs compared", " vs ".join(runs), "all prompts, mean ± 95% CI")
    path = out_dir / "compare_runs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def make_all(clips, summary, fits, run, out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [
        nfe_vs(summary, run, ["sim_ref", "sim_hi_nfe"], out_dir / "nfe_vs_sim.png", "Speaker similarity vs NFE"),
        nfe_vs(summary, run, ["wer", "wer_heard"], out_dir / "nfe_vs_wer.png", "WER vs NFE (lower is better)"),
        nfe_vs_latency(summary, fits, run, out_dir / "nfe_vs_latency.png"),
        latency_vs_quality(summary, run, out_dir / "latency_vs_quality.png"),
        prompt_heatmap(clips, run, out_dir / "prompt_heatmap.png"),
    ]
    return [p for p in made if p]
