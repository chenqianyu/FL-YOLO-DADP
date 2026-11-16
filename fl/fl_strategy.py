"""Strategy layer: any Flower strategy + per-round checkpointing of the
aggregated YOLO model.

The round-number injection and checkpointing are mixed into whichever base
strategy is named on the command line, so FedMedian (the paper's default),
FedAvg, FedTrimmedAvg, Krum, ... all work unchanged.
"""

import logging
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from config import NEEDS_INITIAL_PARAMETERS, FLConfig
from yolo_glue import get_parameters, load_model, save_aggregated_model

logger = logging.getLogger(__name__)


def aggregate_evaluate_metrics(
    results: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Unweighted mean of client mAP50, as in the notebook."""
    if not results:
        return {}
    total = sum(float(m.get("mAP50", 0.0)) for _, m in results)
    return {"mAP50": total / len(results)}


class _SaveModelMixin:
    """Injects `server_round` into client configs and checkpoints each round's
    aggregated model to round{N}/FL_aggregated_round_{N}.pt."""

    def __init__(self, cfg: FLConfig, device, **kwargs):
        self.cfg = cfg
        self._device = device
        self.client_metrics: Dict[str, List[Tuple[int, Dict[str, Scalar]]]] = {}
        super().__init__(**kwargs)

    # ---------------------------------------------------------------- configure
    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        out = super().configure_fit(server_round, parameters, client_manager, **kwargs)
        for _, fit_ins in out:
            if fit_ins.config is None:
                fit_ins.config = {}
            fit_ins.config["server_round"] = server_round
        return out

    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
        out = super().configure_evaluate(
            server_round, parameters, client_manager, **kwargs
        )
        for _, eval_ins in out:
            if eval_ins.config is None:
                eval_ins.config = {}
            eval_ins.config["server_round"] = server_round
        return out

    # ---------------------------------------------------------------- aggregate
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None, {}

        aggregated_parameters, metrics = aggregated
        if aggregated_parameters is None:
            logger.warning("Round %s produced no aggregate; nothing saved", server_round)
            return aggregated

        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        save_aggregated_model(self.cfg, ndarrays, server_round, self._device)
        return aggregated_parameters, metrics

    def evaluate(self, server_round, parameters):
        # None => rely on client-side (distributed) evaluation only.
        return None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        for client_proxy, evaluate_res in results:
            # Key by the agent's own id, not client_proxy.cid: in simulation the
            # proxy cid is a random Ray node id that changes between rounds.
            key = evaluate_res.metrics.get("agent_id", client_proxy.cid)
            self.client_metrics.setdefault(str(key), []).append(
                (server_round, evaluate_res.metrics)
            )
        return aggregated


def build_strategy(cfg: FLConfig, device):
    """Return an instance of `<cfg.strategy>` with round-checkpointing mixed in."""
    base = getattr(fl.server.strategy, cfg.strategy)
    cls = type(f"{cfg.strategy}SaveModel", (_SaveModelMixin, base), {})

    kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.num_clients,
        min_evaluate_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    if cfg.strategy in NEEDS_INITIAL_PARAMETERS:
        seed_net = load_model(cfg, 1, agent_id="server")
        kwargs["initial_parameters"] = fl.common.ndarrays_to_parameters(
            get_parameters(seed_net)
        )
        del seed_net

    return cls(cfg=cfg, device=device, **kwargs)
