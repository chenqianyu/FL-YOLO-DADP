#!/usr/bin/env python3
"""Build the per-agent train/valid/test shards and the pooled set used in the paper.

For each agent, a fixed number of frames is drawn for every cell-type group
(groups are identified by substrings of the frame file names):

    train : 150 per group x 16 groups = 2400 frames
    valid :  15 per group x 16 groups =  240 frames
    test  :  15 per group x 16 groups =  240 frames

and the pooled set (Table 1 "YOLOv8n" benchmark) is the union of the agents'
train and valid splits (7200 / 720 frames).

    python data_prep/make_agent_splits.py \
        --source-images /path/to/detection_set/train/images \
        --source-labels /path/to/detection_set/train/labels \
        --out data/agents --pooled-out data/pooled --seed 0

Train frames are drawn first, then valid/test frames from the remaining pool.
``--allow-split-overlap`` draws valid/test without excluding the train frames.
"""

import argparse
import fnmatch
import os
import random
import shutil

# Cell-type groups, matched with fnmatch as *<pattern>*. Groups overlap
# (e.g. "_mono_" also matches "_mono_act_"); a frame is used at most once.
GROUPS = [
    "_b_cell_", "_eos_", "_cfe", "_cfp", "_mono_", "_mono_act_", "_neu_", "_neu_act_", "_pc",
    "_plt_*_1_*", "_plt_act_", "_plt_agg_", "_rbc_", "_t_cell_", "_wbc_", "_neu_plt_",
]
CLASS_NAMES = ["RBC", "WBC", "PLT"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def write_data_yaml(folder: str, splits) -> None:
    lines = [f"path: {os.path.abspath(folder)}"]
    lines += [f"train: {splits['train']}/images", f"val: {splits['val']}/images"]
    if "test" in splits:
        lines.append(f"test: {splits['test']}/images")
    lines += ["", f"nc: {len(CLASS_NAMES)}", f"names: {CLASS_NAMES}", ""]
    with open(os.path.join(folder, "data.yaml"), "w") as f:
        f.write("\n".join(lines))


def copy_pair(img: str, src_img: str, src_lbl: str, dst: str) -> None:
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
    lbl = os.path.splitext(img)[0] + ".txt"
    if not os.path.isfile(os.path.join(src_lbl, lbl)):
        raise FileNotFoundError(f"label {lbl} not found for image {img}")
    shutil.copy(os.path.join(src_img, img), os.path.join(dst, "images", img))
    shutil.copy(os.path.join(src_lbl, lbl), os.path.join(dst, "labels", lbl))


def draw(rng, pool, used, pattern, n):
    candidates = [f for f in pool if fnmatch.fnmatch(f, f"*{pattern}*") and f not in used]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} frames left for group {pattern!r}, need {n}")
    picked = rng.sample(candidates, n)
    used.update(picked)
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-images", required=True)
    p.add_argument("--source-labels", required=True)
    p.add_argument("--out", default="data/agents")
    p.add_argument("--pooled-out", default=None, help="also build the pooled train/valid set here")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--train-per-group", type=int, default=150)
    p.add_argument("--valid-per-group", type=int, default=15)
    p.add_argument("--test-per-group", type=int, default=15)
    p.add_argument("--seed", type=int, default=None, help="RNG seed (unseeded if omitted)")
    p.add_argument("--allow-split-overlap", action="store_true",
                   help="valid/test draws may reuse frames drawn for train")
    a = p.parse_args()

    rng = random.Random(a.seed)
    pool = sorted(f for f in os.listdir(a.source_images) if f.lower().endswith(IMG_EXTS))
    per_split = {"train": a.train_per_group, "valid": a.valid_per_group, "test": a.test_per_group}

    # Pass 1: train. Pass 2: valid + test.
    used_train = set()
    selection = {k: {} for k in range(a.num_agents)}
    for k in range(a.num_agents):
        selection[k]["train"] = [f for g in GROUPS for f in draw(rng, pool, used_train, g, per_split["train"])]
    used_eval = set() if a.allow_split_overlap else set(used_train)
    for k in range(a.num_agents):
        for split in ("valid", "test"):
            selection[k][split] = [f for g in GROUPS for f in draw(rng, pool, used_eval, g, per_split[split])]

    for k in range(a.num_agents):
        agent_dir = os.path.join(a.out, f"agent_{k}")
        for split, files in selection[k].items():
            for img in files:
                copy_pair(img, a.source_images, a.source_labels, os.path.join(agent_dir, split))
        write_data_yaml(agent_dir, {"train": "train", "val": "valid", "test": "test"})
        print(f"agent_{k}: " + ", ".join(f"{s}={len(v)}" for s, v in selection[k].items()))

    if a.pooled_out:
        for k in range(a.num_agents):
            for split in ("train", "valid"):
                for img in selection[k][split]:
                    copy_pair(img, a.source_images, a.source_labels, os.path.join(a.pooled_out, split))
        write_data_yaml(a.pooled_out, {"train": "train", "val": "valid"})
        print(f"pooled: {a.pooled_out}")

    overlap = {
        s: len(set(f for k in selection for f in selection[k][s]) & used_train) for s in ("valid", "test")
    }
    print(f"frames shared with train: {overlap}")


if __name__ == "__main__":
    main()
