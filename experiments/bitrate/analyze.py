"""Summarize the saved paired outcomes without rerunning training."""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("on_time_pct", "important_on_time_pct", "loss_pct", "delivered_p95_ms")
BOOTSTRAPS = 10000


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def intervals(differences):
    """Paired differences with aligned policies, traces and training seeds."""
    seeds, traces = differences.shape
    rng = np.random.default_rng(8220)
    trace_counts = rng.multinomial(traces, np.full(traces, 1 / traces), BOOTSTRAPS)
    trace_means = trace_counts @ differences.T / traces
    fixed = np.quantile(trace_means.mean(axis=1), [0.025, 0.975])
    seed_counts = rng.multinomial(seeds, np.full(seeds, 1 / seeds), BOOTSTRAPS)
    joint = np.quantile((trace_means * seed_counts).sum(axis=1) / seeds, [0.025, 0.975])
    return [*fixed, *joint]


def plot(summary, contrast, path):
    colors = ["#284D77", "#AF5634", "#267164"]
    rates = [0.75, 1.0, 1.25]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), layout="constrained")
    for policy, label, color in zip(("single_with", "single_without", "fixed"),
                                  ("Q with importance", "Q without importance", "Fixed threshold"), colors):
        vals = [summary[("recheck", "Fluctuating", r, policy)][0] for r in rates]
        axes[0].plot(rates, vals, marker="o", label=label, color=color)
    axes[0].set_ylabel("On-time delivery (% of generated frames)")
    axes[0].set_title("A. Fresh fluctuating traces", loc="left")
    for scenario, color, marker in zip(("Fluctuating", "Sudden drop"), colors, ("o", "s")):
        vals = np.array([contrast[(scenario, r)] for r in rates])
        x = np.array(rates) + (-0.008 if scenario == "Fluctuating" else 0.008)
        axes[1].errorbar(x, vals[:, 0], yerr=[vals[:, 0]-vals[:, 1], vals[:, 2]-vals[:, 0]],
                         marker=marker, color=color, capsize=4, label=scenario)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("B. Mixed minus single training", loc="left")
    axes[1].set_ylabel("On-time change, Q with importance (pp)")
    pairs = (("single_with", "single_without"), ("mixed_with", "mixed_without"),
             ("mixed_with", "single_with"), ("mixed_without", "single_without"))
    for i, pair in enumerate(pairs):
        val, lo, hi = contrast[pair]
        axes[2].errorbar(val, i, xerr=[[val-lo], [hi-val]], fmt="o", color=colors[i // 2], capsize=4)
    axes[2].axvline(0, color="#555555", linewidth=0.8)
    axes[2].set_yticks(range(4), ["Importance, single", "Importance, mixed",
                                 "Training, with importance", "Training, without importance"], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_ylim(3.65, -0.65)
    axes[2].set_title("C. High bitrate, 15 new seeds", loc="left")
    axes[2].set_xlabel("Paired change in on-time delivery (pp)")
    axes[2].grid(axis="x", alpha=0.2)
    for ax in axes[:2]:
        ax.set_xticks(rates, ["0.75×", "1.00×", "1.25×"])
        ax.set_xlabel("Video bitrate relative to training default")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize(output):
    groups = defaultdict(lambda: defaultdict(dict))
    with gzip.open(output / "paired_traces.csv.gz", "rt") as file:
        for row in csv.DictReader(file):
            key = (row["experiment"], row["scenario"], float(row["rate"]), row["policy"])
            layer = groups[key][int(row["seed"])]
            i = int(row["trace"])
            assert i not in layer
            layer[i] = [float(row[m]) for m in METRICS]
    arrays, summary, contrasts, by_seed, selected = {}, {}, [], [], {}
    for key, layer in groups.items():
        expected = {"pilot": 20, "recheck": 120, "training": 100, "confirmation": 100}[key[0]]
        seed_ids = range(5, 20) if key[0] == "confirmation" else range(5)
        assert set(layer) == (set(seed_ids) if key[-1].startswith(("single_", "mixed_")) else {-1})
        assert all(set(rows) == set(range(expected)) for rows in layer.values())
        a = np.array([[layer[s][i] for i in range(expected)] for s in sorted(layer)])
        assert np.isfinite(a).all()
        arrays[key] = a
        summary[key] = a.mean(axis=(0, 1))
    for phase, scenario, rate in sorted({key[:3] for key in groups}):
        k = (phase, scenario, rate)
        pairs = [("single_with", "single_without")]
        if phase in ("training", "confirmation"):
            pairs += [("mixed_with", "single_with"), ("mixed_without", "single_without"),
                      ("mixed_with", "mixed_without")]
        for a, b in pairs:
            diff = arrays[k + (a,)] - arrays[k + (b,)]
            for j, metric in enumerate(METRICS):
                delta = diff[:, :, j]
                bounds = intervals(delta)
                contrasts.append((*k, a, b, metric, delta.mean(), *bounds))
                for seed, value in enumerate(delta.mean(axis=1), start=5 if phase == "confirmation" else 0):
                    by_seed.append((*k, a, b, metric, seed, value))
                if phase == "training" and a == "mixed_with" and b == "single_with" and j == 0:
                    selected[(scenario, rate)] = [delta.mean(), *bounds[2:]]
                if phase == "confirmation" and rate == 1.25 and j == 0:
                    selected[(a, b)] = [delta.mean(), *bounds[2:]]
                if phase == "pilot" and j == 0:
                    assert abs(delta.mean() - 3.95) < 0.015
    keys = ["experiment", "scenario", "rate"]
    write_csv(output / "summary.csv", [*keys, "policy", *METRICS],
              [(*k, *v) for k, v in summary.items()])
    write_csv(output / "contrasts.csv", [*keys, "policy_a", "policy_b", "metric", "a_minus_b",
              "trace_lo", "trace_hi", "trace_seed_lo", "trace_seed_hi"], contrasts)
    write_csv(output / "seed_differences.csv", [*keys, "policy_a", "policy_b", "metric", "seed", "a_minus_b"], by_seed)
    plot(summary, selected, output / "bitrate_comparison.png")


if __name__ == "__main__":
    summarize(Path(__file__).resolve().parent / "results")
