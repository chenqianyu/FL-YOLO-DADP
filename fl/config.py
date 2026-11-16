"""Configuration for federated YOLO training with optional local differential privacy.

Training defaults match the arguments recorded in the paper's runs
(`args.yaml` of every local training job): YOLOv8n, imgsz 384x512, rect,
SGD (lr0 0.01, momentum 0.937, weight decay 5e-4), batch 32, seed 99,
1 local epoch per round, 20 rounds, 3 clients, FedMedian.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

FL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FL_DIR)

# Strategies that build their update from a delta against an initial point and
# therefore need `initial_parameters` supplied explicitly.
NEEDS_INITIAL_PARAMETERS = {"FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"}

DP_MODES = ("none", "ldp", "dadp")


@dataclass
class FLConfig:
    # ---------------------------------------------------------------- identity
    strategy: str = "FedMedian"
    run_name: Optional[str] = None  # auto-derived if None
    project_root: str = os.path.join(REPO_ROOT, "runs", "fl")

    # -------------------------------------------------------------- federation
    num_clients: int = 3
    num_rounds: int = 20
    epochs: int = 1  # local epochs per round

    # ------------------------------------------------------------ model / data
    # Round-1 starting point for every client. In the paper this was a YOLOv8n
    # trained from scratch for 1 epoch on agent_1 (see baselines/train_yolo.py).
    seed_model: str = os.path.join(REPO_ROOT, "weights", "seed_yolov8n_agent1_epoch1.pt")
    # Folder holding one sub-folder per client: <data_root>/agent_<k>/data.yaml
    data_root: str = os.path.join(REPO_ROOT, "data", "agents")
    agent_dir_fmt: str = "agent_{cid}"
    data_yaml_name: str = "data.yaml"

    # -------------------------------------------------- ultralytics train args
    imgsz: List[int] = field(default_factory=lambda: [384, 512])
    rect: bool = True
    optimizer: str = "SGD"
    batch: int = 32
    seed: int = 99
    workers: int = 0
    plots: bool = False
    device: str = "cuda:0"

    # ------------------------------------------------------------ flower / ray
    num_cpus: float = 1
    num_gpus: float = 0.1  # GPU fraction per client; 0.1 => up to 10 at once

    # ---------------------------------------------------- differential privacy
    # none : plain federated learning (FedAvg-YOLO / FedMedian-YOLO)
    # ldp  : static clipping + Gaussian noise on each client's update
    # dadp : dynamic adaptive DP -- EMA clipping norm (Eq. 6) + noise schedule (Eq. 5)
    dp: str = "none"
    noise_multiplier: float = 1e-4  # sigma_0
    clipping_norm: float = 1.0      # C (ldp) / initial C^(0) (dadp)
    clip_momentum: float = 0.9      # theta in Eq. 6
    dp_alpha: float = 0.5           # Eq. 5, exploration floor
    dp_omega: float = 0.25          # Eq. 5, exploration decay rate
    dp_beta: float = 0.2            # Eq. 5, exploitation slope
    dp_delta: float = 1e-5          # target delta for the RDP accountant
    # DADP variant: clip = EMA of the update norm, noise scaled by C^(r) (fl/dp.py).
    dp_strict: bool = False

    # -------------------------------------------------------------- behaviour
    overwrite: bool = False
    # Report each client's real training-set size instead of 1 (FedAvg weighting).
    real_num_examples: bool = False
    # Validate without passing data= (ultralytics then uses the dataset recorded
    # in the checkpoint's training arguments).
    legacy_val: bool = False

    # ------------------------------------------------------------- derivations
    @property
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        if self.dp == "none":
            return f"{self.strategy}_{self.num_clients}agents_round{self.num_rounds}_epoch{self.epochs}"
        name = (
            f"{self.strategy}_{self.dp.upper()}_round{self.num_rounds}_epoch{self.epochs}"
            f"_nm{self.noise_multiplier:g}_cn{self.clipping_norm:g}"
        )
        return name + ("_strict" if self.dp_strict else "")

    @property
    def project_path(self) -> str:
        return os.path.join(self.project_root, self.resolved_run_name)

    def agent_dir(self, cid) -> str:
        return os.path.join(self.data_root, self.agent_dir_fmt.format(cid=cid))

    def data_yaml(self, cid) -> str:
        # Absolute: ultralytics resolves relative dataset roots against its own
        # DATASETS_DIR setting, not the working directory.
        return os.path.abspath(os.path.join(self.agent_dir(cid), self.data_yaml_name))

    def round_dir(self, round_num: int) -> str:
        return os.path.join(self.project_path, f"round{round_num}")

    def agent_out_dir(self, round_num: int, agent_id) -> str:
        return os.path.join(self.round_dir(round_num), f"agent{agent_id}")

    def aggregated_ckpt(self, round_num: int) -> str:
        return os.path.join(self.round_dir(round_num), f"FL_aggregated_round_{round_num}.pt")

    def dp_dir(self) -> str:
        return os.path.join(self.project_path, "dp")

    def validate(self) -> None:
        """Fail fast on problems that would otherwise waste a 20-round run."""
        problems = []
        if not os.path.isfile(self.seed_model):
            problems.append(f"seed_model not found: {self.seed_model}")
        for cid in range(self.num_clients):
            y = self.data_yaml(cid)
            if not os.path.isfile(y):
                problems.append(f"data.yaml not found for client {cid}: {y}")
        if self.strategy not in _known_strategies():
            problems.append(
                f"unknown strategy {self.strategy!r}; "
                f"available: {', '.join(sorted(_known_strategies()))}"
            )
        if self.dp not in DP_MODES:
            problems.append(f"--dp must be one of {DP_MODES}, got {self.dp!r}")
        if self.dp != "none":
            if self.noise_multiplier < 0:
                problems.append("--noise-multiplier must be >= 0")
            if self.clipping_norm <= 0:
                problems.append("--clipping-norm must be > 0")
            if not 0.0 <= self.clip_momentum < 1.0:
                problems.append("--clip-momentum must be in [0, 1)")
        if problems:
            raise SystemExit("Configuration problems:\n  - " + "\n  - ".join(problems))


def _known_strategies():
    import flwr as fl

    return {n for n in dir(fl.server.strategy) if not n.startswith("_") and n[0].isupper()}
