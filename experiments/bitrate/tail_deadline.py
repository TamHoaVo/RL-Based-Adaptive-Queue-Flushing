"""Train and evaluate Q-learning with an optional predicted tail-deadline flag."""
import argparse
import csv
import gzip
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from run import EPISODES, RATES, SCENARIOS, load_sim, scaled, training_rates
from analyze import METRICS, intervals, write_csv

HERE = Path(__file__).resolve().parent
SEEDS = range(5, 20)
TRACE_COUNT = 120
TEST_STREAM = 7
POLICIES = ("single_base", "single_tail", "mixed_base", "mixed_tail")
OUTPUT = HERE / "tail_deadline"


def observation(env, state, tail):
    if not tail:
        return state
    # gecko_action returns 1 exactly when the estimated tail delay exceeds 150 ms.
    # Only the prediction is exposed. The agent selects its own action from its table.
    return state + (env.gecko_action(),)


def fit(task):
    seed, diet, tail = task
    sim = load_sim()
    rng = np.random.default_rng(seed)
    schedule = training_rates(seed) if diet == "mixed" else [1.0] * EPISODES
    shape = (3, 3, 3, 2)
    if tail:
        shape += (2,)
    q = np.zeros(shape + (2,))
    for episode, rate in enumerate(schedule):
        trace_seed = np.random.SeedSequence([1, seed, episode])
        trace = sim.make_trace(SCENARIOS[episode % 3], trace_seed)
        env = sim.WirelessQueue(scaled(trace, rate))
        state = observation(env, env.state(), tail)
        epsilon = max(0.05, 0.995 ** episode)
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(2))
            else:
                action = int(np.argmax(q[state]))
            next_state, reward = env.step(action)
            future = 0.0
            if next_state is not None:
                next_state = observation(env, next_state, tail)
                future = np.max(q[next_state])
            index = state + (action,)
            q[index] += 0.10 * (reward + 0.95 * future - q[index])
            if next_state is None:
                break
            state = next_state
    return seed, diet + ("_tail" if tail else "_base"), q


def check_controls(tables):
    # Compare per-trace metrics for all controls against the earlier branch results.
    sim = load_sim()
    trace = scaled(sim.make_trace("Fluctuating", np.random.SeedSequence([6, 0])), 1.25)
    checked = 0
    with gzip.open(HERE / "results/paired_traces.csv.gz", "rt") as file:
        for r in csv.DictReader(file):
            if (r["experiment"], r["scenario"], r["rate"], r["trace"]) != ("confirmation", "Fluctuating", "1.25", "0"):
                continue
            if r["policy"] not in ("mixed_with", "single_with"):
                continue
            key = r["policy"].replace("_with", "_base")
            actions = tables[int(r["seed"]), key].argmax(axis=-1)
            result = sim.evaluate(lambda env: actions[env.state()], [trace])
            np.testing.assert_allclose(result[[3, 4, 2, 1]], [float(r[m]) for m in METRICS], rtol=0, atol=1e-10)
            checked += 1
    assert checked == 2 * len(SEEDS)


def evaluate(task):
    scenario, rate, tables = task
    sim = load_sim()
    traces = [scaled(sim.make_trace(scenario, np.random.SeedSequence([TEST_STREAM, i])), rate)
              for i in range(TRACE_COUNT)]
    rows = []

    def record(seed, policy, decide):
        for i, trace in enumerate(traces):
            metrics = sim.evaluate(decide, [trace])
            rows.append((scenario, rate, seed, policy, i, *metrics[[3, 4, 2, 1]]))

    for (seed, policy), table in tables.items():
        tail = policy.endswith("_tail")
        actions = table.argmax(axis=-1)
        record(seed, policy, lambda env: actions[observation(env, env.state(), tail)])
    record(-1, "fixed", lambda env: env.gecko_action())
    return rows


def run(workers, output):
    output.mkdir(exist_ok=True, parents=True)
    jobs = [(s, d, t) for s in SEEDS for d in ("single", "mixed") for t in (False, True)]
    with ProcessPoolExecutor(workers) as pool:
        tables = {(s, p): q for s, p, q in pool.map(fit, jobs)}
        check_controls(tables)
        np.savez_compressed(output / "q_tables.npz", **{f"{p}_seed_{s}": q for (s, p), q in tables.items()})
        print("Trained 60 models and verified 30 original-state controls.", flush=True)
        raw = output / "per_trace.csv.gz"
        with gzip.open(raw.with_suffix(".tmp"), "wt", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("scenario", "rate", "seed", "policy", "trace", *METRICS))
            tasks = [(sc, r, tables) for sc in SCENARIOS for r in RATES]
            for task, rows in zip(tasks, pool.map(evaluate, tasks)):
                writer.writerows(rows)
                print(task[:2], len(rows), flush=True)
        raw.with_suffix(".tmp").replace(raw)
    summarize(output)


def summarize(output):
    arrays = {}
    for sc in SCENARIOS:
        for rate in RATES:
            for p in (*POLICIES, "fixed"):
                n = 1 if p == "fixed" else len(SEEDS)
                arrays[sc, rate, p] = np.full((n, TRACE_COUNT, len(METRICS)), np.nan)
    with gzip.open(output / "per_trace.csv.gz", "rt") as file:
        for row in csv.DictReader(file):
            policy, seed, trace = row["policy"], int(row["seed"]), int(row["trace"])
            assert 0 <= trace < TRACE_COUNT and (seed == -1 if policy == "fixed" else seed in SEEDS)
            cell = arrays[row["scenario"], float(row["rate"]), policy][0 if policy == "fixed" else seed-SEEDS.start, trace]
            assert np.isnan(cell).all()
            cell[:] = [float(row[m]) for m in METRICS]
    assert all(np.isfinite(a).all() for a in arrays.values())
    write_csv(output / "metrics.csv", ["scenario", "rate", "policy", *METRICS],
              [(*k, *a.mean(axis=(0, 1))) for k, a in arrays.items()])
    gaps, seed_results = [], []
    for sc in SCENARIOS:
        for rate in RATES:
            fixed = arrays[sc, rate, "fixed"][:, :, 0]
            for diet in ("single", "mixed"):
                base = arrays[sc, rate, diet+"_base"][:, :, 0]
                tail = arrays[sc, rate, diet+"_tail"][:, :, 0]
                before, after = fixed - base, fixed - tail
                gain = tail - base
                lo, hi = intervals(gain)[2:]
                after_lo, after_hi = intervals(after)[2:]
                gaps.append((sc, rate, diet, before.mean(), after.mean(), gain.mean(), lo, hi, after_lo, after_hi))
                for s, b, a in zip(SEEDS, before.mean(axis=1), after.mean(axis=1)):
                    seed_results.append((sc, rate, diet, s, b, a, b-a))
    write_csv(output / "gaps.csv", ["scenario", "rate", "training", "old_gap_pp", "new_gap_pp",
              "gap_reduction_pp", "reduction_lo", "reduction_hi", "new_gap_lo", "new_gap_hi"], gaps)
    write_csv(output / "by_seed.csv", ["scenario", "rate", "training", "seed", "old_gap_pp", "new_gap_pp",
              "gap_reduction_pp"], seed_results)
    plot(output, arrays)


def plot(output, arrays):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5), sharey=True, layout="constrained")
    for ax, diet in zip(axes, ("single", "mixed")):
        for label, policy, color in (("Original state", diet+"_base", "#335981"),
                                     ("With deadline flag", diet+"_tail", "#AB4F32"),
                                     ("Fixed threshold", "fixed", "#24715F")):
            values = [arrays["Fluctuating", r, policy][:, :, 0].mean() for r in RATES]
            ax.plot(RATES, values, marker="o", label=label, color=color)
        ax.set_title(f"{diet.capitalize()}-rate training")
        ax.set_xticks(RATES, ["0.75×", "1.00×", "1.25×"])
        ax.set_xlabel("Video bitrate relative to default")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Frames delivered within 150 ms (%)")
    axes[1].legend(frameon=False)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output)
    else:
        run(args.workers, args.output)
