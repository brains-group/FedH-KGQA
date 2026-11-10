#!/usr/bin/env python3
"""Figure 3 of the FedH-KGQA paper (JIIS): robustness of FedH-KGQA.

Generates fig_fedh_robustness.pdf and a 600 dpi PNG, sized to the sn-jnl text width (5.15 in)
so that the text prints at its true size. The panels vary the cross-client ratio, the number
of clients and the question-type skew.

Numbers are the seed-0 test results reported in the paper. No error bars are drawn, because
these ablations use one seed. N = 3 uses the seed-0 values of the main experiment.

Requirements:
    python -m pip install matplotlib numpy seaborn
Usage:
    python plot_fedh_figures.py --outdir ./figures
In Overleaf:
    \\includegraphics[width=\\textwidth]{fig_fedh_robustness.png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

TEXTWIDTH = 5.15  # inches; \textwidth of the compiled sn-jnl manuscript

# Palette shared with Figs. 1 and 2 of the paper (checked for color-vision deficiency).
# Line style and marker also differ, so the figures read in grayscale too.
INK, MUTED = "#1F2D3D", "#4F5B69"
COLORS = {
    "FedH-KGQA": "#2E7D4F",                            # green, as in Figs. 1 and 2
    "Isolated training + relay execution": "#2563B8",  # blue
    "Isolated": "#B5473A",                             # red, as the failing case in Fig. 1
}
STYLES = {
    "FedH-KGQA": dict(color=COLORS["FedH-KGQA"], lw=1.7, ls="-", marker="o", ms=4.2, zorder=4),
    "Isolated training + relay execution": dict(color=COLORS["Isolated training + relay execution"], lw=1.4,
                                                ls="--", marker="s", ms=3.8, zorder=3),
    "Isolated": dict(color=COLORS["Isolated"], lw=1.4, ls=":", marker="^", ms=4.6, zorder=2),
}
METHODS = ["FedH-KGQA", "Isolated training + relay execution", "Isolated"]

ROBUSTNESS = {
    "MetaQA 3-hop": {
        "ratio": {
            "x": [0.61, 0.75],  # reached ratios; lower ratios are not reachable with three clients
            "FedH-KGQA": [0.9974, 0.9974],
            "Isolated training + relay execution": [0.9768, 0.9475],
            "Isolated": [0.8481, 0.7265],
        },
        "clients": {
            "x": [3, 5, 10],
            "FedH-KGQA": [0.9974, 0.9978, 0.9985],
            "Isolated": [0.8481, 0.8254, 0.5941],
            # Isolated training + relay execution was not run at 5 and 10 clients.
        },
        "skew": {
            "x": [0, 1, 2],
            "FedH-KGQA": [0.9974, 0.9969, 0.9554],
            "Isolated training + relay execution": [0.9768, 0.9091, 0.7517],
            "Isolated": [0.8481, 0.8048, 0.6632],
        },
    },
    "PQ-3H": {
        "ratio": {
            "x": [0.26, 0.50, 0.72],
            "FedH-KGQA": [1.0000, 0.9890, 0.9840],
            "Isolated training + relay execution": [0.9649, 0.8967, 0.7570],
            "Isolated": [0.7719, 0.5563, 0.3833],
        },
        "clients": {
            "x": [3, 5, 10],
            "FedH-KGQA": [0.9890, 1.0000, 0.9923],
            "Isolated": [0.5563, 0.5573, 0.5739],
        },
        "skew": {
            "x": [0, 1, 2],
            "FedH-KGQA": [0.9890, 0.9962, 0.9559],
            "Isolated training + relay execution": [0.8967, 0.7889, 0.6372],
            "Isolated": [0.5563, 0.4743, 0.3388],
        },
    },
}

def setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        # Figs. 1 and 2 print their labels at 6.5 pt and their titles at 7.4 pt; match them.
        "font.size": 6.5,
        "axes.titlesize": 7.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": "#C9D3DE",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#E6EAEF",
        "grid.linewidth": 0.7,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def format_axis(ax: plt.Axes, show_y: bool) -> None:
    ax.set_ylim(0.30, 1.04)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", length=0, pad=3)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=True)
    if not show_y:
        ax.tick_params(axis="y", labelleft=False)


def plot_line(ax: plt.Axes, x, y, method: str) -> None:
    st = STYLES[method]
    ax.plot(x, y, markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=1.3, clip_on=False, **st)


def figure_robustness(outdir: Path) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.15), sharey=True)
    titles = ["(a) Cross-client ratio", "(b) Number of clients", "(c) Question-type skew"]
    keys = ["ratio", "clients", "skew"]
    for col, title in enumerate(titles):
        axs[0, col].set_title(title, pad=7)

    for row, (dataset, experiments) in enumerate(ROBUSTNESS.items()):
        for col, key in enumerate(keys):
            ax = axs[row, col]
            res = experiments[key]
            x = np.asarray(res["x"], dtype=float)
            format_axis(ax, show_y=(col == 0))
            if col == 0:
                ax.set_ylabel(f"{dataset}\nHits@1")
            relay = "Isolated training + relay execution"
            if relay in res:
                # Gap between the two relay-based settings = what federated training adds.
                ax.fill_between(x, res[relay], res["FedH-KGQA"], color=COLORS["FedH-KGQA"], alpha=0.13,
                                lw=0, zorder=1)
            for m in reversed(METHODS):
                if m in res:
                    plot_line(ax, x, res[m], m)
            if key == "ratio":
                ax.set_xlim(0.20, 0.80)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v:.2f}" for v in x])
                if row == 0:
                    ax.axvspan(0.20, 0.575, color="#F5F7FA", lw=0, zorder=0)
                    ax.text(0.3875, 0.62, "not reachable\nwith three\nclients", ha="center",
                            va="center", fontsize=6.5, color=MUTED)
                if row == 1:
                    ax.set_xlabel("reached ratio $\\rho$")
            elif key == "clients":
                ax.set_xlim(2.3, 10.7)
                ax.set_xticks([3, 5, 10])
                if row == 1:
                    ax.set_xlabel("number of clients $N$")
            else:
                ax.set_xlim(-0.25, 2.25)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(["3 of 3", "2 of 3", "1 of 3"])
                if row == 1:
                    ax.set_xlabel("clients that train each type")

    handles = [Line2D([0], [0], markerfacecolor="white", markeredgecolor=STYLES[m]["color"], markeredgewidth=1.5,
                      label=m, **STYLES[m]) for m in METHODS]
    handles.append(Patch(facecolor=COLORS["FedH-KGQA"], alpha=0.13, lw=0, label="gain from federated training"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005), frameon=False,
               handlelength=2.4, handletextpad=0.5, columnspacing=1.3, fontsize=6.5)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.17, wspace=0.10, hspace=0.38)
    export(fig, outdir, "fig_fedh_robustness")


def export(fig: plt.Figure, outdir: Path, stem: str) -> None:
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        print(f"Created: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Directory for the generated PDF and PNG figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    setup_style()
    figure_robustness(args.outdir)


if __name__ == "__main__":
    main()
