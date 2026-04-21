import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── Plot theme ─────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render text (requires local TeX install)
        "font.family": "serif",  # Use serif font
        "font.serif": ["Computer Modern Roman"],  # Standard Overleaf/LaTeX font
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.facecolor": "white",  # White background
        "axes.facecolor": "white",  # White plot area
        "axes.edgecolor": "black",  # Black spines
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "#cccccc",  # Light gray grid
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.labelcolor": "black",
    }
)

# Colour palette
C_NORMAL = "#3fb950"  # green  — normal flows
C_ATTACK = "#f85149"  # red    — attack flow
C_DETECT = "#ff9500"  # amber  — detection event
C_PROB_HIGH = "#f85149"
C_PROB_LOW = "#3fb950"
C_PHASE_N = "#3fb95015"  # phase shading
C_PHASE_A = "#f8514920"
C_PHASE_R = "#58a6ff15"

FLOW_COLOURS = {
    "h1": "#58a6ff",
    "h2": "#bc8cff",
    "h3": "#3fb950",
    "h_attack": "#f85149",
    "web_srv": "#e3b341",
    "db_srv": "#ffa657",
}

# ── Helper for LaTeX Strings ───────────────────────────────────────────────────


def tex_esc(text):
    """Escapes characters that would break LaTeX rendering (like underscores)."""
    return str(text).replace("_", r"\_")


# ── Demo data generator ────────────────────────────────────────────────────────


def make_demo_data():
    """
    Generate synthetic experiment data that realistically represents what the
    experiment produces, based on observed controller log values throughout
    this project.
    """
    rng = np.random.default_rng(42)
    phase = {"normal_s": 30, "attack_s": 60, "recovery_s": 30}
    total = phase["normal_s"] + phase["attack_s"] + phase["recovery_s"]

    def classify_events(
        t_start, t_end, src_name, dst_name, base_prob, attack_prob_fn=None, interval=2.0
    ):
        """Generate [CLASSIFY] events at 2-second intervals."""
        events = []
        t = t_start + interval
        while t < t_end:
            is_attack_flow = src_name == "h_attack"
            if attack_prob_fn:
                prob = float(np.clip(attack_prob_fn(t) + rng.normal(0, 0.02), 0, 1))
            else:
                prob = float(np.clip(base_prob + rng.normal(0, 0.01), 0, 1))
            pkts = (
                int(rng.uniform(50, 200) * interval)
                if not is_attack_flow
                else int(rng.uniform(5000, 15000) * interval)
            )
            byts_s = pkts * rng.uniform(1200, 1500)
            events.append(
                {
                    "type": "classify",
                    "rel_s": round(t, 2),
                    "src": (
                        "00:00:00:00:00:20" if is_attack_flow else "00:00:00:00:00:01"
                    ),
                    "dst": "00:00:00:00:00:10",
                    "src_name": src_name,
                    "dst_name": dst_name,
                    "pkts": pkts,
                    "dur_s": round(t, 2),
                    "byts_s": round(byts_s, 1),
                    "bwd_bytes": int(pkts * rng.uniform(1200, 1500) * 0.8),
                    "prob": prob,
                    "is_attack_flow": is_attack_flow,
                }
            )
            t += interval + rng.uniform(-0.1, 0.1)
        return events

    def make_run(model_key, detects):
        events = []
        # Normal flows — always low probability
        for src, dst in [("h1", "web_srv"), ("h2", "db_srv"), ("h3", "web_srv")]:
            events += classify_events(0, total, src, dst, base_prob=0.03)

        # Attack flow — only present during attack + recovery phases
        attack_start = phase["normal_s"]
        recovery_start = attack_start + phase["attack_s"]

        if model_key == "mininet":
            # Mininet model correctly ramps up quickly
            def prob_fn(t):
                if t < attack_start:
                    return 0.02
                ramp = min(1.0, (t - attack_start) / 6.0)
                return 0.85 * ramp + 0.08

            events += classify_events(
                attack_start,
                recovery_start,
                "h_attack",
                "web_srv",
                base_prob=0.02,
                attack_prob_fn=prob_fn,
            )
            # After attack stopped: prob drops toward 0 as no new pkts
            events += classify_events(
                recovery_start, total, "h_attack", "web_srv", base_prob=0.02
            )
            # Detection event
            first_above = next(
                (e for e in events if e["is_attack_flow"] and e["prob"] >= 0.50), None
            )
            if first_above:
                events.append(
                    {
                        "type": "detection",
                        "rel_s": first_above["rel_s"],
                        "src": "00:00:00:00:00:20",
                        "dst": "00:00:00:00:00:10",
                        "src_name": "h_attack",
                        "dst_name": "web_srv",
                        "prob": first_above["prob"],
                    }
                )
        else:
            # InSDN model — does not detect Mininet-scale ICMP flood
            events += classify_events(
                attack_start, total, "h_attack", "web_srv", base_prob=0.00
            )

        return {
            "config_key": model_key,
            "label": (
                "Mininet Model (Emulation-Calibrated)"
                if model_key == "mininet"
                else "InSDN Model (Real-Hardware Dataset)"
            ),
            "winner_file": (
                "model_rf_mininet.joblib"
                if model_key == "mininet"
                else "model_dt.joblib"
            ),
            "phase_times": {
                "normal_start": 0.0,
                "attack_start": float(attack_start),
                "recovery_start": float(recovery_start),
                "end": float(total),
            },
            "events": sorted(events, key=lambda e: e["rel_s"]),
        }

    return {
        "run_timestamp": "demo",
        "phase_durations": phase,
        "runs": [make_run("insdn", False), make_run("mininet", True)],
    }


# ── Data helpers ───────────────────────────────────────────────────────────────


def extract_flow_series(events, src_name):
    """Return (times, probs, byts_s, pkts) for a given source host."""
    rows = [e for e in events if e["type"] == "classify" and e["src_name"] == src_name]
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    rows.sort(key=lambda e: e["rel_s"])
    t = np.array([r["rel_s"] for r in rows])
    prob = np.array([r["prob"] for r in rows])
    byts_s = np.array([r["byts_s"] for r in rows])
    pkts = np.array([r["pkts"] for r in rows])
    return t, prob, byts_s, pkts


def first_detection(events):
    dets = [e for e in events if e["type"] == "detection"]
    return dets[0] if dets else None


# ── Phase shading helper ───────────────────────────────────────────────────────


def shade_phases(ax, pt, total):
    """Shade Normal / Attack / Recovery regions on an axis."""
    ax.axvspan(0, pt["attack_start"], color=C_PHASE_N, zorder=0)
    ax.axvspan(pt["attack_start"], pt["recovery_start"], color=C_PHASE_A, zorder=0)
    ax.axvspan(pt["recovery_start"], total, color=C_PHASE_R, zorder=0)
    # Phase labels at top
    y_top = ax.get_ylim()[1]
    kw = dict(fontsize=7.5, alpha=0.65, va="top", transform=ax.get_xaxis_transform())
    ax.text(
        (0 + pt["attack_start"]) / 2,
        0.98,
        "NORMAL",
        color="darkgreen",
        ha="center",
        **kw,
    )
    ax.text(
        (pt["attack_start"] + pt["recovery_start"]) / 2,
        0.98,
        "ATTACK",
        color="darkred",
        ha="center",
        **kw,
    )
    ax.text(
        (pt["recovery_start"] + total) / 2,
        0.98,
        "RECOVERY",
        color="darkblue",
        ha="center",
        **kw,
    )


def mark_detection(ax, det, yval=None, label=True):
    """Draw a vertical orange line at the detection event."""
    if det is None:
        return
    ax.axvline(det["rel_s"], color=C_DETECT, linewidth=1.8, linestyle="--", zorder=5)
    if label:
        ax.text(
            det["rel_s"] + 0.5,
            yval if yval else ax.get_ylim()[1] * 0.85,
            f"DETECTED\n$t={det['rel_s']:.0f}$s\n$p={det['prob']:.2f}$",
            color="black",
            fontsize=7.5,
            va="top",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_DETECT, lw=1),
        )


# ── Figure 1: side-by-side timeline comparison ────────────────────────────────


def fig_timeline_comparison(data, out_path):
    runs = {r["config_key"]: r for r in data["runs"]}
    pd = data["phase_durations"]
    total = pd["normal_s"] + pd["attack_s"] + pd["recovery_s"]
    models = ["insdn", "mininet"]

    fig = plt.figure(figsize=(18, 10), facecolor="white")

    # Brief Title
    fig.suptitle(
        "Timeline Comparison", fontsize=14, fontweight="bold", color="black", y=0.98
    )

    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        hspace=0.05,
        wspace=0.12,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.07,
    )

    for col, mk in enumerate(models):
        if mk not in runs:
            continue
        run = runs[mk]
        pt = run["phase_times"]
        evts = run["events"]
        det = first_detection(evts)

        inner = gridspec.GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=outer[col],
            hspace=0.08,
            height_ratios=[3, 2, 1.5],
        )
        ax_prob = fig.add_subplot(inner[0])
        ax_bytes = fig.add_subplot(inner[1], sharex=ax_prob)
        ax_pkts = fig.add_subplot(inner[2], sharex=ax_prob)

        # ── Probability panel ─────────────────────────────────────────────────
        shade_phases(ax_prob, pt, total)
        for src in ["h1", "h2", "h3"]:
            t, prob, _, _ = extract_flow_series(evts, src)
            if len(t):
                ax_prob.plot(
                    t,
                    prob,
                    color=FLOW_COLOURS[src],
                    lw=1.2,
                    alpha=0.7,
                    label=tex_esc(src),
                )

        t_atk, prob_atk, _, _ = extract_flow_series(evts, "h_attack")
        if len(t_atk):
            ax_prob.plot(
                t_atk,
                prob_atk,
                color=C_ATTACK,
                lw=2.0,
                zorder=4,
                label=tex_esc("h_attack"),
            )
            ax_prob.fill_between(
                t_atk, 0, prob_atk, color=C_ATTACK, alpha=0.12, zorder=3
            )

        ax_prob.axhline(
            0.5,
            color=C_DETECT,
            linewidth=1,
            linestyle=":",
            alpha=0.8,
            label="threshold (0.50)",
        )
        mark_detection(ax_prob, det, yval=0.5)
        ax_prob.set_ylim(-0.02, 1.08)
        ax_prob.set_ylabel("Attack Probability", fontsize=9)
        ax_prob.set_title(
            tex_esc(run["label"]),
            fontsize=10.5,
            fontweight="bold",
            pad=8,
            color="black",
        )
        ax_prob.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
        plt.setp(ax_prob.get_xticklabels(), visible=False)

        # ── Byte rate panel ───────────────────────────────────────────────────
        shade_phases(ax_bytes, pt, total)
        for src in ["h1", "h2", "h3"]:
            t, _, byts_s, _ = extract_flow_series(evts, src)
            if len(t):
                ax_bytes.plot(
                    t, byts_s / 1e6, color=FLOW_COLOURS[src], lw=1.2, alpha=0.7
                )
        if len(t_atk):
            _, _, byts_atk, _ = extract_flow_series(evts, "h_attack")
            ax_bytes.plot(t_atk, byts_atk / 1e6, color=C_ATTACK, lw=2.0, zorder=4)
        mark_detection(ax_bytes, det, label=False)
        ax_bytes.set_ylabel("Flow Rate (MB/s)", fontsize=9)
        plt.setp(ax_bytes.get_xticklabels(), visible=False)

        # ── Packet count panel ────────────────────────────────────────────────
        shade_phases(ax_pkts, pt, total)
        for src in ["h1", "h2", "h3"]:
            t, _, _, pkts = extract_flow_series(evts, src)
            if len(t):
                ax_pkts.bar(
                    t,
                    pkts,
                    width=1.6,
                    color=FLOW_COLOURS[src],
                    alpha=0.55,
                    label=tex_esc(src),
                )
        if len(t_atk):
            _, _, _, pkts_atk = extract_flow_series(evts, "h_attack")
            ax_pkts.bar(
                t_atk, pkts_atk, width=1.6, color=C_ATTACK, alpha=0.65, zorder=4
            )
        mark_detection(ax_pkts, det, label=False)
        ax_pkts.set_ylabel("Pkts / interval", fontsize=9)
        ax_pkts.set_xlabel("Time (seconds)", fontsize=9)
        ax_pkts.set_xlim(0, total)

        # Annotate detection latency if present
        if det:
            latency = det["rel_s"] - pt["attack_start"]
            ax_prob.annotate(
                f"$\\Delta t = {latency:.0f}$s",
                xy=(det["rel_s"], 0.5),
                xytext=(det["rel_s"] - 12, 0.60),
                color="black",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C_DETECT, lw=1.2),
            )
        elif mk == "insdn":
            ax_prob.text(
                0.5,
                0.5,
                "No detection\n(domain shift:\nMininet $\\ll$ InSDN rates)",
                transform=ax_prob.transAxes,
                fontsize=9,
                color="black",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85),
            )

    # Shared legend
    handles = [
        mpatches.Patch(color=FLOW_COLOURS["h1"], label=tex_esc("h1 (normal)")),
        mpatches.Patch(color=FLOW_COLOURS["h2"], label=tex_esc("h2 (normal)")),
        mpatches.Patch(color=FLOW_COLOURS["h3"], label=tex_esc("h3 (normal)")),
        mpatches.Patch(color=C_ATTACK, label=tex_esc("h_attack")),
        Line2D([0], [0], color=C_DETECT, lw=1.5, ls="--", label="Detection event"),
        Line2D([0], [0], color=C_DETECT, lw=1, ls=":", label="Threshold (0.50)"),
        mpatches.Patch(color=C_PHASE_N, alpha=0.6, label="Normal phase"),
        mpatches.Patch(color=C_PHASE_A, alpha=0.6, label="Attack phase"),
        mpatches.Patch(color=C_PHASE_R, alpha=0.6, label="Recovery phase"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=9,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 2 & 3: per-model deep-dive ─────────────────────────────────────────


def fig_model_detail(run, out_path, pd):
    total = pd["normal_s"] + pd["attack_s"] + pd["recovery_s"]
    pt = run["phase_times"]
    evts = run["events"]
    det = first_detection(evts)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        facecolor="white",
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 1.5], "hspace": 0.07},
    )

    # Brief Title
    fig.suptitle(tex_esc(run["label"]), fontsize=12, fontweight="bold", color="black")

    ax_prob, ax_bytes, ax_bwd, ax_pkts = axes

    # ── Panel 1: probability ──────────────────────────────────────────────────
    shade_phases(ax_prob, pt, total)
    for src, colour in FLOW_COLOURS.items():
        t, prob, _, _ = extract_flow_series(evts, src)
        if not len(t):
            continue
        lw = 2.2 if src == "h_attack" else 1.2
        ax_prob.plot(
            t,
            prob,
            color=colour,
            lw=lw,
            label=tex_esc(src),
            zorder=4 if src == "h_attack" else 2,
        )
        if src == "h_attack":
            ax_prob.fill_between(t, 0, prob, color=colour, alpha=0.15)
    ax_prob.axhline(0.5, color=C_DETECT, lw=1, ls=":", alpha=0.8)
    mark_detection(ax_prob, det)
    ax_prob.set_ylim(-0.02, 1.12)
    ax_prob.set_ylabel("Attack Probability")
    ax_prob.legend(ncol=3, fontsize=8, loc="upper right", framealpha=0.9)

    # ── Panel 2: byte rate ────────────────────────────────────────────────────
    shade_phases(ax_bytes, pt, total)
    for src, colour in FLOW_COLOURS.items():
        t, _, byts_s, _ = extract_flow_series(evts, src)
        if not len(t):
            continue
        ax_bytes.plot(t, byts_s / 1e6, color=colour, lw=1.5 if src == "h_attack" else 1)
    mark_detection(ax_bytes, det, label=False)
    ax_bytes.set_ylabel("Byte Rate (MB/s)")

    # ── Panel 3: backward bytes (key feature for attack detection) ────────────
    shade_phases(ax_bwd, pt, total)
    atk_rows = [e for e in evts if e["type"] == "classify" and e["is_attack_flow"]]
    if atk_rows:
        t_bwd = np.array([e["rel_s"] for e in atk_rows])
        bwd = np.array([e["bwd_bytes"] for e in atk_rows])
        ax_bwd.fill_between(
            t_bwd,
            0,
            bwd / 1e6,
            color=C_ATTACK,
            alpha=0.4,
            label=tex_esc("h_attack bwd bytes"),
        )
        ax_bwd.plot(t_bwd, bwd / 1e6, color=C_ATTACK, lw=1.5)
    mark_detection(ax_bwd, det, label=False)
    ax_bwd.set_ylabel("Bwd Bytes (MB)\n[key feature]")

    # ── Panel 4: packets ──────────────────────────────────────────────────────
    shade_phases(ax_pkts, pt, total)
    for src, colour in FLOW_COLOURS.items():
        t, _, _, pkts = extract_flow_series(evts, src)
        if not len(t):
            continue
        ax_pkts.bar(t, pkts, width=1.5, color=colour, alpha=0.6)
    mark_detection(ax_pkts, det, label=False)
    ax_pkts.set_ylabel("Pkts / interval")
    ax_pkts.set_xlabel("Experiment Time (seconds)")
    ax_pkts.set_xlim(0, total)

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 4: probability heatmap over time ────────────────────────────────────


def fig_probability_heatmap(data, out_path):
    runs = data["runs"]
    pd = data["phase_durations"]
    total = pd["normal_s"] + pd["attack_s"] + pd["recovery_s"]
    bins = np.arange(0, total + 2, 2)
    hosts = ["h1", "h2", "h3", "h_attack"]

    fig, axes = plt.subplots(1, len(runs), figsize=(16, 5), facecolor="white")
    if len(runs) == 1:
        axes = [axes]

    # Brief Title
    fig.suptitle(
        "Attack Probability Heatmap", fontsize=13, fontweight="bold", color="black"
    )

    for ax, run in zip(axes, runs):
        evts = run["events"]
        pt = run["phase_times"]
        grid = np.zeros((len(hosts), len(bins) - 1))
        for r, src in enumerate(hosts):
            rows = [e for e in evts if e["type"] == "classify" and e["src_name"] == src]
            for e in rows:
                idx = int(e["rel_s"] // 2)
                if 0 <= idx < grid.shape[1]:
                    grid[r, idx] = max(grid[r, idx], e["prob"])

        im = ax.imshow(
            grid,
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            extent=[0, total, len(hosts) - 0.5, -0.5],
        )
        ax.set_yticks(range(len(hosts)))
        ax.set_yticklabels([tex_esc(h) for h in hosts])
        ax.set_xlabel("Time (s)")
        ax.set_title(tex_esc(run["label"]), fontsize=10, color="black")

        # Phase lines
        for t_mark, label in [
            (pt["attack_start"], "Attack\nstart"),
            (pt["recovery_start"], "Attack\nstop"),
        ]:
            ax.axvline(t_mark, color="black", lw=1.5, ls="--", alpha=0.6)

        det = first_detection(evts)
        if det:
            ax.axvline(det["rel_s"], color=C_DETECT, lw=2, ls="-", alpha=0.9)

        plt.colorbar(im, ax=ax, label="P(Attack)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 5: detection latency bar ────────────────────────────────────────────


def fig_detection_latency(data, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    # Brief Title
    fig.suptitle("Detection Latency", fontsize=13, fontweight="bold", color="black")

    bars_data = []
    for run in data["runs"]:
        det = first_detection(run["events"])
        pt = run["phase_times"]
        if det:
            latency = det["rel_s"] - pt["attack_start"]
            bars_data.append((run["label"], latency, det["prob"], True))
        else:
            bars_data.append((run["label"], None, None, False))

    labels = [tex_esc(b[0]) for b in bars_data]
    latencies = [b[1] if b[1] is not None else 0 for b in bars_data]
    detected = [b[3] for b in bars_data]
    colours = [C_NORMAL if d else "black" for d in detected]

    bars = ax.barh(labels, latencies, color=colours, edgecolor="black", height=0.45)
    for bar, (lbl, lat, prob, det) in zip(bars, bars_data):
        if det:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"${lat:.1f}$s  ($p={prob:.3f}$)",
                va="center",
                color="black",
                fontsize=9,
            )
        else:
            ax.text(
                2,
                bar.get_y() + bar.get_height() / 2,
                "No detection — domain shift",
                va="center",
                color="black",
                fontsize=9,
                fontstyle="italic",
            )

    ax.set_xlabel("Seconds from attack start to first detection")
    ax.set_xlim(0, max(latencies or [1]) * 1.4 + 10)
    ax.axvline(0, color="black", lw=1)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="IDS Experiment Visualizer")
    parser.add_argument("--results", help="Path to events.json from experiment.py")
    parser.add_argument(
        "--demo", action="store_true", help="Generate demo figures with synthetic data"
    )
    parser.add_argument("--out-dir", help="Output directory (default: same as results)")
    args = parser.parse_args()

    if args.demo:
        data = make_demo_data()
        out_dir = args.out_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "demo"
        )
    elif args.results:
        with open(args.results) as fp:
            data = json.load(fp)
        out_dir = args.out_dir or os.path.dirname(args.results)
    else:
        parser.print_help()
        sys.exit("\nError: provide --results <path> or --demo")

    os.makedirs(out_dir, exist_ok=True)
    pd = data["phase_durations"]
    runs_by_key = {r["config_key"]: r for r in data["runs"]}

    print(f"\nGenerating figures → {out_dir}")

    fig_timeline_comparison(
        data,
        os.path.join(out_dir, "fig1_timeline_comparison.png"),
    )

    for key, fname in [
        ("insdn", "fig2_insdn_detail.png"),
        ("mininet", "fig3_mininet_detail.png"),
    ]:
        if key in runs_by_key:
            fig_model_detail(
                runs_by_key[key],
                os.path.join(out_dir, fname),
                pd,
            )

    fig_probability_heatmap(
        data,
        os.path.join(out_dir, "fig4_probability_heatmap.png"),
    )

    fig_detection_latency(
        data,
        os.path.join(out_dir, "fig5_detection_latency.png"),
    )

    print(f"\nAll figures saved to: {out_dir}")
    print("Figures:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(out_dir, f)) // 1024
            print(f"  {f}  ({size} KB)")


if __name__ == "__main__":
    main()
