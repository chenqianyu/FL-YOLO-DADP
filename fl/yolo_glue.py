"""YOLO <-> Flower glue: model loading, local training, validation, parameter I/O."""

import glob
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from config import FLConfig

logger = logging.getLogger(__name__)

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def ensure_directory_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_device(cfg: FLConfig) -> torch.device:
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(cfg.device)


def load_model(cfg: FLConfig, round_num: int, agent_id=None) -> YOLO:
    """Round 1 starts from the seed checkpoint, later rounds from the previous
    round's aggregated model. The weights are overwritten right after by the
    parameters the server sends; loading the file gives ultralytics a model
    object with the right architecture and args attached."""
    path = cfg.seed_model if round_num <= 1 else cfg.aggregated_ckpt(round_num - 1)
    logger.info("Loading %s for agent %s", path, agent_id)
    net = YOLO(path)
    # Stop net.train() from reloading the on-disk weights over the parameters
    # the server just handed us.
    net.ckpt_path = ""
    if hasattr(net.model, "ckpt_path"):
        net.model.ckpt_path = ""
    return net


def train(cfg: FLConfig, net: YOLO, data: str, round_num: int, agent_id) -> None:
    """One round of local training. Ultralytics writes the local model to
    <run>/round{N}/agent{K}/train_round{N}_agent{K}/weights/{best,last}.pt."""
    save_path = cfg.agent_out_dir(round_num, agent_id)
    ensure_directory_exists(save_path)
    net.ckpt_path = ""
    if hasattr(net.model, "ckpt_path"):
        net.model.ckpt_path = ""
    net.train(
        data=data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        rect=cfg.rect,
        optimizer=cfg.optimizer,
        seed=cfg.seed,
        batch=cfg.batch,
        plots=cfg.plots,
        device=cfg.device,
        project=save_path,
        name=f"train_round{round_num}_agent{agent_id}",
        exist_ok=False,
        workers=cfg.workers,
        pretrained=False,  # do not let ultralytics reset the weights
        resume=False,
    )
    logger.info("Training completed for agent %s in round %s", agent_id, round_num)


def test(cfg: FLConfig, net: YOLO, round_num: int, agent_id, data: str) -> float:
    """mAP50 of the current global model on this client's own validation split."""
    val_path = cfg.agent_out_dir(round_num, agent_id)
    ensure_directory_exists(val_path)
    kwargs = dict(project=val_path, imgsz=cfg.imgsz, rect=cfg.rect, batch=cfg.batch, plots=cfg.plots)
    if not cfg.legacy_val:
        kwargs["data"] = data  # this client's own validation split
    metrics = net.val(**kwargs)
    return float(metrics.box.map50)


def set_parameters(net: YOLO, parameters: List[np.ndarray], device) -> None:
    ref = net.model.state_dict()
    state_dict = {
        k: torch.tensor(v, dtype=ref[k].dtype).to(device) for k, v in zip(ref.keys(), parameters)
    }
    net.model.load_state_dict(state_dict, strict=True)


def get_parameters(net: YOLO) -> List[np.ndarray]:
    return [p.cpu().numpy() for p in net.model.state_dict().values()]


def save_checkpoint(net: YOLO, path: str) -> None:
    """Write ``net.model`` as an ultralytics checkpoint, in full precision.

    ``YOLO.save()`` is avoided on purpose. In ultralytics 8.1.x it writes the
    checkpoint dict that was loaded at ``YOLO(path)`` time, i.e. the weights
    *before* any later change. In 8.3.x it keeps a stale ``ema`` entry from that
    dict, and ``YOLO(path)`` loads ``ema`` in preference to ``model``.
    """
    import ultralytics

    ckpt = dict(net.ckpt or {})
    ckpt.update(
        {
            "model": deepcopy(net.model).cpu().float(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "date": datetime.now().isoformat(),
            "version": ultralytics.__version__,
        }
    )
    ensure_directory_exists(os.path.dirname(path))
    torch.save(ckpt, path)


def save_aggregated_model(cfg: FLConfig, parameters: List[np.ndarray], round_num: int, device) -> str:
    model = load_model(cfg, round_num, agent_id="server")
    set_parameters(model, parameters, device)
    out = cfg.aggregated_ckpt(round_num)
    save_checkpoint(model, out)
    logger.info("Aggregated model saved at %s", out)
    return out


def count_train_images(cfg: FLConfig, cid) -> int:
    """Real training-set size for a client, used when cfg.real_num_examples."""
    img_dir = os.path.join(cfg.agent_dir(cid), "train", "images")
    n = sum(len(glob.glob(os.path.join(img_dir, e))) for e in IMG_EXTS)
    if n == 0:
        logger.warning("No training images found under %s; reporting 1", img_dir)
        return 1
    return n
