#!/usr/bin/env python3
"""Evaluate trained detectors on a held-out split and write a Table-1-style CSV.

    python evaluation/evaluate.py --data data/agents/agent_0/data.yaml --split test --out results/table1.csv \
        --model "YOLOv8n=runs/baselines/benchmark_yolov8n/weights/best.pt" \
        --model "Encoder-decoder=runs/baselines/synthetic_encoder-decoder/weights/best.pt" \
        --model "GANs=runs/baselines/synthetic_GANs/weights/best.pt" \
        --model "DDPM=runs/baselines/synthetic_DDPM/weights/best.pt" \
        --model "Controlnet-img2img=runs/baselines/synthetic_controlnet_img2img/weights/best.pt" \
        --model "FedAvg-YOLO=runs/fl/FedAvg_3agents_round20_epoch1/round20/FL_aggregated_round_20.pt" \
        --model "FedMedian-YOLO=runs/fl/FedMedian_3agents_round20_epoch1/round20/FL_aggregated_round_20.pt" \
        --model "FL-YOLO-DADP=runs/fl/FedMedian_DADP_round20_epoch1_nm0.0001_cn1/round20/FL_aggregated_round_20.pt"

Columns: macro precision, macro recall, mAP50 over all classes, AP50 per class,
mAP50-95, and inference time per image (ms). Per-class values are AP at IoU 0.5
(``box.ap50``), not AP@0.5:0.95 (``box.maps``).
"""

import argparse
import csv
import os

from ultralytics import YOLO


def parse_model_arg(s: str):
    if "=" not in s:
        return os.path.splitext(os.path.basename(s))[0], s
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def evaluate(name: str, weights: str, a) -> dict:
    model = YOLO(weights)
    r = model.val(
        data=os.path.abspath(a.data),
        split=a.split,
        imgsz=a.imgsz,
        rect=True,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        plots=False,
        project=a.project,
        name=name.replace(" ", "_"),
        exist_ok=True,
        verbose=False,
    )
    box = r.box
    row = {
        "model": name,
        "precision": box.mp,
        "recall": box.mr,
        "AP50_all": box.map50,
    }
    per_class = {int(c): float(ap) for c, ap in zip(box.ap_class_index, box.ap50)}
    for idx, cls_name in r.names.items():
        row[f"AP50_{cls_name}"] = per_class.get(int(idx), float("nan"))
    row["mAP50_95"] = box.map
    row["inference_ms"] = r.speed.get("inference")
    row["weights"] = weights
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="data.yaml with a `test:` entry")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--model", action="append", required=True, help="NAME=path/to/weights.pt (repeatable)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[384, 512])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0, help="dataloader worker processes")
    p.add_argument("--project", default="runs/eval")
    p.add_argument("--out", default="results/table1.csv")
    a = p.parse_args()

    rows = [evaluate(*parse_model_arg(m), a) for m in a.model]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = [c for c in rows[0] if c not in ("model", "weights")]
    print(f"{'model':<22}" + "".join(f"{c:>12}" for c in cols))
    for r in rows:
        print(f"{r['model']:<22}" + "".join(
            f"{r[c]:>12.3f}" if isinstance(r[c], float) else f"{str(r[c]):>12}" for c in cols))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
