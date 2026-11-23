"""Replicate the pilot and compare training diets. Run from any directory with Python 3.10+."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RATES = (0.75, 1.0, 1.25)
SCENARIOS = ("Stable", "Fluctuating", "Sudden drop")
FIELDS = ["experiment", "scenario", "rate", "policy", "seed", "trace",
          "on_time_pct", "important_on_time_pct", "loss_pct", "delivered_p95_ms"]
N_RECHECK, N_COMPARE = 120, 100
N_SEEDS, EPISODES = 5, 600


def load_sim():
    path = ROOT / "RLv2.ipynb"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "7e1b9cf71a6d7820f38c027a7e3bbd9f25af42f9bb6b8a10b2a6a808dbf031a6":
        raise RuntimeError("Use the RLv2.ipynb from upstream commit b79fb68.")
    module = types.ModuleType("queue_simulator_v2")
    sys.modules[module.__name__] = module
    code = json.loads(path.read_text())["cells"][1]["source"]
    exec(compile("".join(code), str(path), "exec"), module.__dict__)
    return module


def scaled(trace, factor):
    sizes, important, rates = trace
    return np.rint(sizes * factor).astype(int), important, rates


def training_rates(seed):
    """Balance all rates within each nine episodes and each scenario to within one."""
    rng = np.random.default_rng(np.random.SeedSequence([5, seed]))
    schedule = []
    while len(schedule) < EPISODES:
        columns = rng.permutation(3)
        for offset in rng.permutation(3):
            schedule.extend(RATES[(s + offset) % 3] for s in columns)
    schedule = schedule[:EPISODES]
    assert all(schedule.count(r) == 200 for r in RATES)
    assert all(66 <= schedule[s::3].count(r) <= 67 for s in range(3) for r in RATES)
    return schedule


def fit(task):
    seed, flag = task
    sim = load_sim()
    single, _ = sim.train(EPISODES, seed, flag)
    if seed < N_SEEDS:
        with ZipFile(ROOT / "gecko_results.zip") as archive:
            saved = np.load(io.BytesIO(archive.read(f"q_{flag}_seed_{seed}.npy")))
        np.testing.assert_array_equal(single, saved)
    original_trace = sim.make_trace
    schedule = iter(training_rates(seed))

    def mixed_trace(scenario, trace_seed):
        return scaled(original_trace(scenario, trace_seed), next(schedule))

    sim.make_trace = mixed_trace
    mixed, _ = sim.train(EPISODES, seed, flag)
    assert next(schedule, None) is None
    return seed, flag, single, mixed


def simulate(task):
    phase, scenario, rate, tables = task
    sim = load_sim()
    stream, count = {"pilot": (2, 20), "recheck": (3, N_RECHECK),
                     "training": (4, N_COMPARE), "confirmation": (6, 100)}[phase]
    # Reuse each base trace across rates and policies to pair comparisons.
    traces = [scaled(sim.make_trace(scenario, np.random.SeedSequence([stream, i])), rate)
              for i in range(count)]
    rows = []

    def record(label, seed, policy, factory=None):
        for i, trace in enumerate(traces):
            metrics = sim.evaluate(policy, [trace], factory)
            rows.append((phase, scenario, rate, label, seed, i, *metrics[[3, 4, 2, 1]]))

    for seed, flag, single, mixed in tables:
        for diet, table in (("single", single), ("mixed", mixed)):
            if phase in ("pilot", "recheck") and diet == "mixed":
                continue
            actions = np.argmax(table, axis=-1)
            record(f"{diet}_{'with' if flag else 'without'}", seed,
                   lambda env: actions[sim.observe(env.state(), flag)])
    record("fixed", -1, lambda env: env.gecko_action())
    record("fifo", -1, lambda env: sim.KEEP)
    if phase != "pilot":
        for atomic, label in ((False, "gecko_selective"), (True, "gecko_atomic")):
            record(label, -1, None, lambda: sim.GeckoAB(atomic))
    return rows


def run(workers, output):
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        tables = list(pool.map(fit, [(s, f) for s in range(N_SEEDS) for f in (True, False)]))
        print("All ten single-rate retrainings exactly matched the archived Q-tables.", flush=True)
        tasks = [("pilot", "Fluctuating", 1.25, tables)]
        tasks += [("recheck", "Fluctuating", r, tables) for r in RATES]
        tasks += [("training", s, r, tables) for s in SCENARIOS for r in RATES]
        fresh = list(pool.map(fit, [(s, f) for s in range(5, 20) for f in (True, False)]))
        tasks += [("confirmation", "Fluctuating", r, fresh) for r in (1.0, 1.25)]
        target = output / "paired_traces.csv.gz"
        with gzip.open(target.with_suffix(".tmp"), "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            for task, rows in zip(tasks, pool.map(simulate, tasks)):
                writer.writerows(rows)
                print(*task[:3], len(rows), flush=True)
        target.with_suffix(".tmp").replace(target)
    from analyze import summarize
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.workers, args.output)
