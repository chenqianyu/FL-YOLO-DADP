#!/usr/bin/env python3
"""Federated YOLOv8 training with optional local differential privacy (LDP / DADP).

Examples
--------
    # FL-YOLO-DADP as in Table 1: FedMedian, sigma_0 = 1e-4, C = 1
    python fl/run_fl.py --strategy FedMedian --dp dadp --noise-multiplier 1e-4 --clipping-norm 1

    # Vanilla federated baselines (Table 1)
    python fl/run_fl.py --strategy FedAvg
    python fl/run_fl.py --strategy FedMedian

    # Static LDP (Fig. 3)
    python fl/run_fl.py --strategy FedAvg --dp ldp --noise-multiplier 1e-3 --clipping-norm 10

    # Smoke test: 1 round, 2 clients
    python fl/run_fl.py --num-rounds 1 --num-clients 2 --run-name smoke --num-gpus 0.45

Output layout
-------------
    <project_root>/<run_name>/
        round{N}/agent{K}/train_round{N}_agent{K}/   ultralytics train run (local model, results.csv)
        round{N}/agent{K}/val*/                      global model validated on agent K's val split
        round{N}/FL_aggregated_round_{N}.pt          global model after round N
        dp/agent{K}.jsonl, dp/privacy.json           per-round DP log and epsilon (DP runs only)
        metrics.csv                                  per-round global-model mAP50 per client
        run_config.json                              exact configuration of this run
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys

FL_DIR = os.path.dirname(os.path.abspath(__file__))
if FL_DIR not in sys.path:
    sys.path.insert(0, FL_DIR)
# Ray workers import these modules too.
os.environ["PYTHONPATH"] = FL_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import flwr as fl  # noqa: E402
import torch  # noqa: E402

from config import DP_MODES, FLConfig  # noqa: E402
from dp import summarize_privacy  # noqa: E402
from fl_client import make_client_fn  # noqa: E402
from fl_strategy import build_strategy  # noqa: E402
from yolo_glue import ensure_directory_exists, resolve_device  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> FLConfig:
    d = FLConfig()
    p = argparse.ArgumentParser(
        description="Federated YOLO training (Flower) with optional LDP / DADP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("experiment")
    g.add_argument("--strategy", default=d.strategy, help="any class in flwr.server.strategy")
    g.add_argument("--run-name", default=d.run_name, help="output folder name (auto if omitted)")
    g.add_argument("--project-root", default=d.project_root)
    g.add_argument("--num-clients", type=int, default=d.num_clients)
    g.add_argument("--num-rounds", type=int, default=d.num_rounds)
    g.add_argument("--epochs", type=int, default=d.epochs, help="local epochs per round")
    g.add_argument("--seed-model", default=d.seed_model, help="round-1 starting checkpoint")
    g.add_argument("--data-root", default=d.data_root, help="folder with agent_<k>/data.yaml")

    g = p.add_argument_group("local training (passed to ultralytics)")
    g.add_argument("--batch", type=int, default=d.batch)
    g.add_argument("--imgsz", type=int, nargs="+", default=d.imgsz)
    g.add_argument("--optimizer", default=d.optimizer)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--workers", type=int, default=d.workers)
    g.add_argument("--device", default=d.device)
    g.add_argument("--plots", action="store_true", default=d.plots)

    g = p.add_argument_group("differential privacy")
    g.add_argument("--dp", choices=DP_MODES, default=d.dp)
    g.add_argument("--noise-multiplier", type=float, default=d.noise_multiplier, help="sigma_0")
    g.add_argument("--clipping-norm", type=float, default=d.clipping_norm, help="C (ldp) / C^(0) (dadp)")
    g.add_argument("--clip-momentum", type=float, default=d.clip_momentum, help="theta in Eq. 6")
    g.add_argument("--dp-alpha", type=float, default=d.dp_alpha, help="alpha in Eq. 5")
    g.add_argument("--dp-omega", type=float, default=d.dp_omega, help="omega in Eq. 5")
    g.add_argument("--dp-beta", type=float, default=d.dp_beta, help="beta in Eq. 5")
    g.add_argument("--dp-delta", type=float, default=d.dp_delta, help="delta for the RDP accountant")
    g.add_argument("--dp-strict", action="store_true",
                   help="DADP variant: clip = EMA of the update norm, noise scaled by C^(r) "
                        "(see fl/dp.py)")

    g = p.add_argument_group("runtime")
    g.add_argument("--num-gpus", type=float, default=d.num_gpus, help="GPU fraction per client")
    g.add_argument("--num-cpus", type=float, default=d.num_cpus)
    g.add_argument("--overwrite", action="store_true", help="delete an existing run directory first")
    g.add_argument("--real-num-examples", action="store_true",
                   help="report true shard sizes instead of 1 (changes FedAvg weighting)")
    g.add_argument("--legacy-val", action="store_true",
                   help="validate without data= (uses the dataset recorded in the checkpoint)")

    a = vars(p.parse_args())
    return FLConfig(**{k: v for k, v in a.items()})


def prepare_output_dir(cfg: FLConfig) -> None:
    path = cfg.project_path
    if os.path.exists(path):
        if not cfg.overwrite:
            raise SystemExit(
                f"Run directory already exists:\n  {path}\n"
                "Pass --overwrite to delete it, or --run-name to pick another name."
            )
        logger.warning("Removing existing run directory %s", path)
        shutil.rmtree(path)
    ensure_directory_exists(path)
    with open(os.path.join(path, "run_config.json"), "w") as f:
        json.dump({**vars(cfg), "resolved_run_name": cfg.resolved_run_name}, f, indent=2)


def write_metrics_csv(cfg: FLConfig, history, strategy) -> str:
    per_round = dict(history.metrics_distributed.get("mAP50", []))
    cids = sorted(strategy.client_metrics, key=lambda c: (0, int(c)) if c.lstrip("-").isdigit() else (1, c))
    lookup = {c: {rnd: m.get("mAP50") for rnd, m in strategy.client_metrics[c]} for c in cids}
    out = os.path.join(cfg.project_path, "metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "mAP50_aggregated"] + [f"mAP50_client_{c}" for c in cids])
        for rnd in sorted(set(per_round) | {r for d in lookup.values() for r in d}):
            w.writerow([rnd, per_round.get(rnd, "")] + [lookup[c].get(rnd, "") for c in cids])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = parse_args()
    cfg.validate()
    prepare_output_dir(cfg)
    device = resolve_device(cfg)
    torch.cuda.empty_cache()

    logger.info("Run directory : %s", cfg.project_path)
    logger.info("Strategy      : %s, %d clients, %d rounds x %d local epoch(s)",
                cfg.strategy, cfg.num_clients, cfg.num_rounds, cfg.epochs)
    if cfg.dp == "none":
        logger.info("Privacy       : none")
    else:
        logger.info("Privacy       : %s, sigma_0=%g, C=%g, theta=%g%s", cfg.dp, cfg.noise_multiplier,
                    cfg.clipping_norm, cfg.clip_momentum, " (strict Eqs. 6-8)" if cfg.dp_strict else "")
    if cfg.legacy_val:
        logger.info("Validation    : without data= (dataset recorded in the checkpoint)")

    strategy = build_strategy(cfg, device)
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(cfg),
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
        ray_init_args={
            "ignore_reinit_error": True,
            "runtime_env": {"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
        },
    )

    print("\nGlobal-model mAP50 (mean over clients' val splits):")
    for rnd, map50 in history.metrics_distributed.get("mAP50", []):
        print(f"  round {rnd:2d}: {map50:.4f}")
    print(f"\nMetrics : {write_metrics_csv(cfg, history, strategy)}")
    print(f"Model   : {cfg.aggregated_ckpt(cfg.num_rounds)}")

    summary = summarize_privacy(cfg)
    if summary:
        print(f"Privacy : epsilon_nominal={summary.get('epsilon_nominal')}, "
              f"epsilon_effective={summary.get('epsilon_effective')} at delta={cfg.dp_delta} "
              f"(details in {os.path.join(cfg.dp_dir(), 'privacy.json')})")


if __name__ == "__main__":
    main()
