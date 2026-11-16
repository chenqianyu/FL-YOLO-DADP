#!/usr/bin/env python3
"""Fig. 3: mAP50 over federated rounds for FedAvg/FedMedian x LDP/DADP, one panel
per (clipping norm, noise multiplier), against the local-only benchmark.

    python evaluation/plot_fl_curves.py --runs-root runs/fl --out results/fig3.png \
        --benchmark runs/baselines/local_agent{0,1,2}_epoch20/results.csv

Run folders are looked up by the names fl/run_fl.py gives them, e.g.
``FedMedian_DADP_round20_epoch1_nm0.0001_cn1``.

``--source`` selects what is plotted per round:

* ``local`` (default, as in the paper): mean over agents of the mAP50 that
  ultralytics logged at the end of each agent's local training in that round,
  on the agent's own validation split.
* ``global``: mean over agents of the global model's mAP50 on each agent's
  validation split, from the run's metrics.csv.

The benchmark curve is the per-epoch mean of the local-only runs passed via
``--benchmark`` (one YOLOv8n per agent, trained on that agent's data only).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = [("FedAvg", "LDP"), ("FedAvg", "DADP"), ("FedMedian", "LDP"), ("FedMedian", "DADP")]
MAP_COL = "metrics/mAP50(B)"


def read_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def local_curve(run_dir, num_clients, num_rounds):
    out = []
    for r in range(1, num_rounds + 1):
        vals = []
        for k in range(num_clients):
            df = read_csv(f"{run_dir}/round{r}/agent{k}/train_round{r}_agent{k}/results.csv")
            if df is not None and MAP_COL in df.columns and len(df):
                vals.append(float(df[MAP_COL].iloc[-1]))
        out.append(np.mean(vals) if vals else np.nan)
    return out


def global_curve(run_dir, num_rounds):
    df = read_csv(f"{run_dir}/metrics.csv")
    if df is None:
        return [np.nan] * num_rounds
    s = df.set_index("round")["mAP50_aggregated"]
    return [float(s.get(r, np.nan)) for r in range(1, num_rounds + 1)]


def benchmark_curve(paths, num_rounds):
    curves = []
    for p in paths:
        df = read_csv(p)
        if df is None:
            print(f"warning: benchmark csv not found: {p}")
            continue
        curves.append(df[MAP_COL].to_numpy()[:num_rounds])
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs/fl")
    p.add_argument("--benchmark", nargs="*", default=[], help="results.csv of the local-only runs")
    p.add_argument("--noise-multipliers", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--clipping-norms", type=float, nargs="+", default=[1, 10])
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--source", choices=["local", "global"], default="local")
    p.add_argument("--out", default="results/fig3.png")
    a = p.parse_args()

    rounds = np.arange(1, a.num_rounds + 1)
    bench = benchmark_curve(a.benchmark, a.num_rounds)
    nr, nc = len(a.noise_multipliers), len(a.clipping_norms)
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 4.2 * nr), squeeze=False)

    for i, nm in enumerate(a.noise_multipliers):
        for j, cn in enumerate(a.clipping_norms):
            ax = axes[i][j]
            for strategy, dp in METHODS:
                run = os.path.join(
                    a.runs_root,
                    f"{strategy}_{dp}_round{a.num_rounds}_epoch{a.epochs}_nm{nm:g}_cn{cn:g}",
                )
                if not os.path.isdir(run):
                    print(f"missing run: {run}")
                    continue
                y = (local_curve(run, a.num_clients, a.num_rounds) if a.source == "local"
                     else global_curve(run, a.num_rounds))
                ax.plot(rounds, y, marker="o", label=f"{strategy} {dp}")
            if bench is not None:
                ax.plot(np.arange(1, len(bench) + 1), bench, color="grey", marker="x",
                        markersize=5, label=f"Benchmark (Avg {len(a.benchmark)} agents)")
            ax.set_title(f"Clipping Norm = {cn:g}, Noise Multiplier = {nm:g}")
            ax.set_xlabel("Federated Rounds")
            ax.set_ylabel("mAP50")
            ax.set_xticks(rounds)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
