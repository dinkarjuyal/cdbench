"""Generate CDBench paper figures from hardcoded experimental results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "3B": "#aaaaaa",
    "3B+fs": "#888888",
    "7B": "#4878cf",
    "7B+fs": "#4878cf",
    "7B+fs+RL": "#e87722",
    "32B+fs": "#2ca02c",
    "Qwen3-8B": "#9467bd",
    "235B MoE+fs": "#d62728",
    "repair": "#ff7f0e",
}

MARKERS = {
    "3B": "v",
    "3B+fs": "^",
    "7B": "s",
    "7B+fs": "D",
    "7B+fs+RL": "o",
    "32B+fs": "*",
    "Qwen3-8B": "P",
    "235B MoE+fs": "X",
}


# ── Figure 1: Degradation curves ──────────────────────────────────────────────

def fig_degradation():
    fig, ax = plt.subplots(figsize=(7, 4.2))

    data = {
        "3B (no fs)":    {1: 40,  3: 17,  5: 8},
        "3B+fs":         {1: 30,  3: 10,  5: 2},
        "7B (no fs)":    {1: 60,  3: 3,   5: 4},
        "7B+fs":         {1: 70,  3: 30,  5: 40},
        "7B+fs+RL":      {1: 60,  3: 47,  5: 16},
        "32B+fs":        {1: 100, 3: 93,  5: 68,  7: 77, 10: 52},
    }

    style = {
        "3B (no fs)":    dict(color="#aaaaaa", ls=":",   marker="v", lw=1.2),
        "3B+fs":         dict(color="#888888", ls=":",   marker="^", lw=1.2),
        "7B (no fs)":    dict(color="#4878cf", ls="--",  marker="s", lw=1.2),
        "7B+fs":         dict(color="#4878cf", ls="-",   marker="D", lw=1.4),
        "7B+fs+RL":      dict(color="#e87722", ls="-",   marker="o", lw=2.0),
        "32B+fs":        dict(color="#2ca02c", ls="-",   marker="*", lw=2.2, ms=10),
    }

    for label, pts in data.items():
        xs = sorted(pts.keys())
        ys = [pts[x] for x in xs]
        kw = style[label]
        ms = kw.pop("ms", 7)
        ax.plot(xs, ys, label=label, markersize=ms, **kw)

    ax.set_xlabel("Bug count $N$", fontsize=12)
    ax.set_ylabel("Functional accuracy (%)", fontsize=12)
    ax.set_title("CDBench degradation curves", fontsize=13)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_ylim(-5, 108)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "degradation_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "degradation_curves.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved degradation_curves.{pdf,png}")


# ── Figure 2: Iterative repair ─────────────────────────────────────────────────

def fig_iterative():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ns = [3, 5, 7]

    # Round-0 and best-of-3 for 7B+fs+RL
    r0   = [40, 32, 33]
    best = [47, 43, 44]

    # 32B+fs ceiling at same N values
    ceiling = [93, 68, 77]

    ax.plot(ns, r0,      color="#e87722", ls="--", marker="o", lw=1.8, ms=7,
            label="7B+fs+RL  round 0")
    ax.plot(ns, best,    color="#e87722", ls="-",  marker="o", lw=2.2, ms=7,
            label="7B+fs+RL  best-of-3")
    ax.fill_between(ns, r0, best, color="#e87722", alpha=0.12)

    ax.plot(ns, ceiling, color="#2ca02c", ls="-",  marker="*", lw=2.0, ms=10,
            label="32B+fs  (ceiling)")

    ax.set_xlabel("Bug count $N$", fontsize=12)
    ax.set_ylabel("Functional accuracy (%)", fontsize=12)
    ax.set_title("Iterative repair flattens the degradation cliff", fontsize=12)
    ax.set_xticks(ns)
    ax.set_ylim(20, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "iterative_repair.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "iterative_repair.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved iterative_repair.{pdf,png}")


# ── Figure 3: Repair-conditioned training ─────────────────────────────────────

def fig_repair_training():
    fig, ax = plt.subplots(figsize=(6, 3.8))

    ns = [1, 3, 5]
    baseline_r0   = [60, 40, 32]
    baseline_best = [60, 47, 43]
    repair_r0     = [50, 58, 33]
    repair_best   = [50, 60, 46]

    x = np.arange(len(ns))
    w = 0.18
    ax.bar(x - 1.5*w, baseline_r0,   w, label="Baseline — round 0",     color="#4878cf", alpha=0.6)
    ax.bar(x - 0.5*w, baseline_best, w, label="Baseline — best",         color="#4878cf")
    ax.bar(x + 0.5*w, repair_r0,     w, label="+Repair train — round 0", color="#e87722", alpha=0.6)
    ax.bar(x + 1.5*w, repair_best,   w, label="+Repair train — best",    color="#e87722")

    ax.set_xlabel("Bug count $N$", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Repair-conditioned GRPO vs. baseline", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylim(0, 75)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "repair_training.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "repair_training.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved repair_training.{pdf,png}")


def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(11, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(1.0, 4.8)
    ax.axis("off")

    ax.text(5.5, 4.55, "CDBench: Corruption Generation and Evaluation Pipeline",
            ha="center", va="top", fontsize=12, fontweight="bold")

    # ── Stage boxes ──────────────────────────────────────────────────────────
    stages = [
        (1.05, "Source\nFiles",   "#e3f2fd", "any library\nwith a test suite"),
        (3.0,  "Proposer\nLLM",   "#fff8e1", "generates candidate\nmutations"),
        (5.0,  "Executor",        "#e8f5e9", "sandbox clone\n+ test suite"),
        (7.0,  "Guide\nLLM",      "#fce4ec", "scores non-triviality\n(1–5 scale)"),
        (9.2,  "Catalog",         "#f3e5f5", "verified\ncorruptions"),
    ]

    for x, title, color, sub in stages:
        ax.add_patch(FancyBboxPatch((x - 0.75, 3.05), 1.5, 1.0,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="#888", lw=1.2))
        ax.text(x, 3.75, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x, 3.2,  sub,   ha="center", va="center", fontsize=7,   color="#444")

    # arrows
    for x1, x2 in [(1.8, 2.25), (3.75, 4.25), (5.75, 6.25), (7.75, 8.45)]:
        ax.annotate("", xy=(x2, 3.55), xytext=(x1, 3.55),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # filter labels under arrows
    for x, label in [(3.0, "single-line mutations\noperator · logic · constant · membership"),
                     (5.0, "discard if no test\nstatus changes"),
                     (7.0, "reject score ≤ 2\n(trivial / obvious)")]:
        ax.text(x, 2.92, label, ha="center", va="top", fontsize=6.8,
                color="#666", style="italic")

    # ── Divider ──────────────────────────────────────────────────────────────
    ax.axhline(y=2.45, xmin=0.02, xmax=0.98, color="#ccc", lw=1, ls="--")
    ax.text(0.25, 2.3, "Controllable\nat evaluation:",
            ha="left", va="top", fontsize=8.5, fontweight="bold", color="#333")

    # ── Parameter boxes ───────────────────────────────────────────────────────
    params = [
        (2.8,  2.05, 2.3,  "Bug count  N",           "#e3f2fd", "N ∈ {1, 3, 5, 7, 10}"),
        (5.5,  2.05, 3.2,  "Spatial distribution",   "#e8f5e9",
         "clustered (all N from 1 file)\nscattered (N from distinct files)"),
        (8.7,  2.05, 3.8,  "Mutation type",           "#fff8e1",
         "operator · logic · constant\nmembership · accumulation"),
    ]

    for x, y, w, title, color, detail in params:
        h = 0.85
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.07",
                                    facecolor=color, edgecolor="#aaa", lw=0.9))
        ax.text(x, y + 0.17, title,  ha="center", va="center", fontsize=8,   fontweight="bold")
        ax.text(x, y - 0.17, detail, ha="center", va="center", fontsize=7,   color="#444")

    fig.savefig(os.path.join(OUT, "pipeline.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved pipeline.{pdf,png}")


if __name__ == "__main__":
    fig_degradation()
    fig_iterative()
    fig_repair_training()
    fig_pipeline()
    print("Done.")
