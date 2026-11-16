#!/usr/bin/env python3
"""Local differential privacy for client model updates: LDP and DADP.

Both mechanisms act on a client's update for round r,

    delta_k = w_k^(r) - w^(r-1)

i.e. every entry of the YOLO ``state_dict`` (weights and BatchNorm running
statistics) flattened into one vector, before it is sent to the server:

    1. clip the whole vector to L2 norm <= clip                 (Eq. 7)
    2. add isotropic Gaussian noise N(0, noise_std^2 I)         (Eq. 8)

LDP uses a static clipping norm C and noise_std = sigma_0 * C.

DADP adapts both over the rounds. The noise multiplier follows Eq. (5) with a
zero-based round index i = r - 1:

    sigma^(r) = sigma_0 * (alpha + (1 - alpha) * exp(-omega * i))   if i <  R // 2
              = sigma_0 * (1 + beta * (i - R // 2))                 otherwise

and the clipping norm follows an exponential moving average h of the update
norm n = ||delta_k|| (Eq. 6), kept per client and initialised to C:

    default       h <- theta*h + (1-theta)*n;  clip = theta*h + (1-theta)*n;  noise_std = sigma^(r) * C
    --dp-strict   h <- theta*h + (1-theta)*n;  clip = h;                      noise_std = sigma^(r) * h

Run this file directly to print the (epsilon, delta) a noise schedule implies:

    python fl/dp.py --dp dadp --noise-multiplier 1e-4 --num-rounds 20
"""

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- schedule
def noise_schedule(
    sigma0: float, round_idx: int, num_rounds: int, alpha: float, omega: float, beta: float
) -> float:
    """Eq. (5). ``round_idx`` is zero-based (round 1 -> 0)."""
    half = num_rounds // 2
    if round_idx < half:
        return sigma0 * (alpha + (1.0 - alpha) * math.exp(-omega * round_idx))
    return sigma0 * (1.0 + beta * (round_idx - half))


def nominal_sigmas(cfg) -> List[float]:
    """Per-round noise multipliers of the configured schedule."""
    if cfg.dp == "ldp":
        return [cfg.noise_multiplier] * cfg.num_rounds
    return [
        noise_schedule(cfg.noise_multiplier, i, cfg.num_rounds, cfg.dp_alpha, cfg.dp_omega, cfg.dp_beta)
        for i in range(cfg.num_rounds)
    ]


# ------------------------------------------------------------------ accounting
def rdp_epsilon(noise_multipliers: Sequence[float], delta: float) -> float:
    """Epsilon of composing one full-participation Gaussian mechanism per round.

    Each entry is noise_std / sensitivity for one round. Returns ``inf`` if any
    round has no noise. Requires ``opacus``.
    """
    if any(z <= 0 for z in noise_multipliers):
        return math.inf
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    for z in noise_multipliers:
        acc.step(noise_multiplier=float(z), sample_rate=1.0)
    return float(acc.get_epsilon(delta=delta))


# ------------------------------------------------------------------ privatizer
class ClientPrivatizer:
    """Clips and noises one client's update; keeps that client's C^(r) on disk.

    Clients are re-instantiated every round in Flower simulation, so the EMA
    state lives in ``<run>/dp/agent<k>_state.json``. Every call appends one line
    to ``<run>/dp/agent<k>.jsonl`` with the norm, clip and noise actually used.
    """

    def __init__(self, cfg, agent_id: int, rng: Optional[np.random.Generator] = None):
        if cfg.dp not in ("ldp", "dadp"):
            raise ValueError(f"ClientPrivatizer needs dp in (ldp, dadp), got {cfg.dp!r}")
        self.cfg = cfg
        self.agent_id = agent_id
        self.rng = rng if rng is not None else np.random.default_rng()
        os.makedirs(cfg.dp_dir(), exist_ok=True)
        self.state_path = os.path.join(cfg.dp_dir(), f"agent{agent_id}_state.json")
        self.log_path = os.path.join(cfg.dp_dir(), f"agent{agent_id}.jsonl")

    # .......................................................... state helpers
    def _load_ema(self) -> float:
        if os.path.isfile(self.state_path):
            with open(self.state_path) as f:
                return float(json.load(f)["ema_norm"])
        return float(self.cfg.clipping_norm)  # C^(0)

    def _save_ema(self, value: float) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"ema_norm": value}, f)
        os.replace(tmp, self.state_path)

    # ................................................................ main op
    def __call__(self, delta: List[np.ndarray], round_num: int) -> List[np.ndarray]:
        cfg = self.cfg
        flat = np.concatenate([d.reshape(-1).astype(np.float32) for d in delta])
        norm = float(np.linalg.norm(flat))
        round_idx = round_num - 1

        if cfg.dp == "ldp":
            sigma = cfg.noise_multiplier
            clip = cfg.clipping_norm
            noise_std = sigma * cfg.clipping_norm
        else:  # dadp
            sigma = noise_schedule(
                cfg.noise_multiplier, round_idx, cfg.num_rounds, cfg.dp_alpha, cfg.dp_omega, cfg.dp_beta
            )
            theta = cfg.clip_momentum
            ema = theta * self._load_ema() + (1.0 - theta) * norm  # Eq. (6)
            self._save_ema(ema)
            if cfg.dp_strict:
                clip = ema
                noise_std = sigma * clip  # Eq. (8)
            else:
                clip = theta * ema + (1.0 - theta) * norm  # second EMA step
                noise_std = sigma * cfg.clipping_norm  # scaled by the initial clipping norm

        if norm > clip:  # Eq. (7)
            flat *= clip / norm
        if noise_std > 0:
            flat += self.rng.normal(0.0, noise_std, size=flat.shape).astype(np.float32)

        out, pos = [], 0
        for d in delta:
            n = int(np.prod(d.shape))
            out.append(flat[pos:pos + n].reshape(d.shape).astype(np.float32))
            pos += n

        record = {
            "round": round_num,
            "sigma": sigma,
            "update_norm": norm,
            "clip": clip,
            "noise_std": noise_std,
            "effective_noise_multiplier": (noise_std / clip) if clip > 0 else 0.0,
            "clipped": norm > clip,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("DP agent %s round %s: %s", self.agent_id, round_num, record)
        return out


# --------------------------------------------------------------- run summary
def summarize_privacy(cfg) -> Optional[Dict]:
    """Collect per-round DP logs of a finished run and compute epsilon two ways.

    * ``epsilon_nominal`` -- the noise schedule sigma^(r) fed to an RDP
      accountant (full participation, sensitivity C).
    * ``epsilon_effective`` -- the noise actually added relative to the
      clipping bound actually applied, taking the worst client per round.
    """
    if cfg.dp == "none":
        return None
    per_round: Dict[int, List[float]] = {}
    for k in range(cfg.num_clients):
        path = os.path.join(cfg.dp_dir(), f"agent{k}.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                per_round.setdefault(int(rec["round"]), []).append(rec["effective_noise_multiplier"])
    effective = [min(per_round[r]) for r in sorted(per_round)]
    summary = {"delta": cfg.dp_delta, "rounds_logged": len(effective)}
    try:
        summary["epsilon_nominal"] = rdp_epsilon(nominal_sigmas(cfg), cfg.dp_delta)
        summary["epsilon_effective"] = rdp_epsilon(effective, cfg.dp_delta) if effective else None
    except ImportError:
        logger.warning("opacus not installed; skipping epsilon computation")
    summary["effective_noise_multiplier_per_round"] = effective
    with open(os.path.join(cfg.dp_dir(), "privacy.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# -------------------------------------------------------------------- CLI
def _main() -> None:
    from types import SimpleNamespace

    p = argparse.ArgumentParser(description="Print the (epsilon, delta) implied by a noise schedule")
    p.add_argument("--dp", choices=["ldp", "dadp"], default="dadp")
    p.add_argument("--noise-multiplier", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.2)
    a = p.parse_args()
    for nm in a.noise_multiplier:
        cfg = SimpleNamespace(
            dp=a.dp, noise_multiplier=nm, num_rounds=a.num_rounds,
            dp_alpha=a.alpha, dp_omega=a.omega, dp_beta=a.beta,
        )
        eps = rdp_epsilon(nominal_sigmas(cfg), a.delta)
        print(f"{a.dp}  sigma_0={nm:g}  rounds={a.num_rounds}  ->  epsilon={eps:.4g}  (delta={a.delta:g})")


if __name__ == "__main__":
    _main()
