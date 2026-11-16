#!/usr/bin/env python3
"""Train YOLOv8n from scratch on one dataset. Used for every non-federated model:

    # Table 1 "YOLOv8n" benchmark: all agents' real images pooled, 20 epochs
    python baselines/train_yolo.py --data data/pooled/data.yaml --epochs 20 --name benchmark_yolov8n

    # Round-1 seed for federated training: agent_1 only, 1 epoch
    python baselines/train_yolo.py --data data/agents/agent_1/data.yaml --epochs 1 \
        --name seed_agent1_epoch1 --export weights/seed_yolov8n_agent1_epoch1.pt

    # Fig. 3 "Benchmark (Avg 3 agents)": one local-only model per agent, 20 epochs
    for k in 0 1 2; do
        python baselines/train_yolo.py --data data/agents/agent_$k/data.yaml --epochs 20 --name local_agent${k}_epoch20
    done

    # Table 1 generative baselines: YOLOv8n on each fully synthetic training set
    python baselines/train_yolo.py --data data/synthetic/GANs/data.yaml --epochs 20 --name synthetic_GANs

All other hyper-parameters are the ones in the paper: batch 32, imgsz 384x512, rect, SGD, lr0 0.01, momentum 0.937,
weight decay 5e-4, seed 99.
"""

import argparse
import os
import shutil

from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ultralytics data.yaml")
    p.add_argument("--name", required=True, help="run name under --project")
    p.add_argument("--model", default="yolov8n.yaml", help="architecture yaml (trained from scratch)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, nargs="+", default=[384, 512])
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=os.path.join(REPO_ROOT, "runs", "baselines"))
    p.add_argument("--export", default=None, help="also copy weights/best.pt to this path")
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(
        data=os.path.abspath(a.data),
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        rect=True,
        optimizer=a.optimizer,
        lr0=a.lr0,
        momentum=a.momentum,
        weight_decay=a.weight_decay,
        seed=a.seed,
        device=a.device,
        project=a.project,
        name=a.name,
    )
    best = os.path.join(a.project, a.name, "weights", "best.pt")
    print(f"best weights: {best}")
    if a.export:
        os.makedirs(os.path.dirname(os.path.abspath(a.export)), exist_ok=True)
        shutil.copy(best, a.export)
        print(f"copied to   : {a.export}")


if __name__ == "__main__":
    main()
