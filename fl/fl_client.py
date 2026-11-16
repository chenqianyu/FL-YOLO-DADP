"""Flower client: one client == one agent == one data shard (agent_<k>/data.yaml).

The model is re-loaded at the start of every fit/evaluate call and then
overwritten with the parameters the server sent. With DP enabled, the client
returns  w^(r-1) + privatize(w_k^(r) - w^(r-1))  instead of its raw weights.
"""

import logging
from typing import Callable, List

import flwr as fl
import numpy as np

from config import FLConfig
from dp import ClientPrivatizer
from yolo_glue import (
    count_train_images,
    get_parameters,
    load_model,
    resolve_device,
    set_parameters,
    test,
    train,
)

logger = logging.getLogger(__name__)


class YoloFlowerClient(fl.client.NumPyClient):
    def __init__(self, cfg: FLConfig, cid: str):
        self.cfg = cfg
        self.agent_id = int(cid)
        self.data_path = cfg.data_yaml(self.agent_id)
        self.device = resolve_device(cfg)
        self.net = None
        self.round_num = 1

    def get_parameters(self, config) -> List[np.ndarray]:
        if self.net is None:
            self.net = load_model(self.cfg, 1, self.agent_id).to(self.device)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        self.round_num = int(config["server_round"])
        self.net = load_model(self.cfg, self.round_num, self.agent_id).to(self.device)
        set_parameters(self.net, parameters, self.device)

        logger.info("round %s, agent %s: local training", self.round_num, self.agent_id)
        train(self.cfg, self.net, data=self.data_path, round_num=self.round_num, agent_id=self.agent_id)
        updated = get_parameters(self.net)

        if self.cfg.dp != "none":
            start = [np.asarray(p, dtype=np.float32) for p in parameters]
            delta = [np.asarray(u, dtype=np.float32) - s for u, s in zip(updated, start)]
            noised = ClientPrivatizer(self.cfg, self.agent_id)(delta, self.round_num)
            updated = [s + d for s, d in zip(start, noised)]

        return updated, self._num_examples(), {}

    def evaluate(self, parameters, config):
        self.round_num = int(config["server_round"])
        self.net = load_model(self.cfg, self.round_num, self.agent_id).to(self.device)
        set_parameters(self.net, parameters, self.device)
        map50 = test(self.cfg, self.net, self.round_num, self.agent_id, data=self.data_path)
        logger.info("round %s, agent %s: global-model mAP50 = %.4f", self.round_num, self.agent_id, map50)
        # Flower's first return value is a loss; report 1 - mAP50 ("lower is better").
        return float(1.0 - map50), self._num_examples(), {"mAP50": map50, "agent_id": self.agent_id}

    def _num_examples(self) -> int:
        if self.cfg.real_num_examples:
            return count_train_images(self.cfg, self.agent_id)
        return 1


def make_client_fn(cfg: FLConfig) -> Callable[[str], fl.client.Client]:
    def client_fn(cid: str) -> fl.client.Client:
        return YoloFlowerClient(cfg, cid).to_client()

    return client_fn
