"""
interfaces.py
=============
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


class PolicyBase(ABC):
    """
    Implementations may be stateless (pure function) or maintain a hidden
    recurrent state that they carry through `act()`.

    All tensors are expected to be *torch* tensors residing on the same device.
    """

    @abstractmethod
    def reset(self, batch: int = 1) -> Any:  # returns initial hidden state
        """Reset internal state between rollouts."""

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        hidden: Any | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Any, Dict[str, Any]]:
        """
        Parameters
        ----------
        obs           : (B, …) batched observation tensor
        hidden        : previous hidden state or None
        deterministic : if True, return the mode instead of sampling

        Returns
        -------
        action  : (B, act_dim) tensor
        new_h   : new hidden state (can be None)
        info    : auxiliary dict (e.g. log-probs, values)
        """


class EnvWrapper(ABC):
    """
    A light Gym-like interface for batched, differentiable environments.
    Extension of Rollout_Agents to calcuate agents steps, and reset
    """

    @abstractmethod
    def reset(self, batch: int = 1) -> torch.Tensor:
        """
        Initialise `batch` independent environments.

        Returns
        -------
        initial observation tensor  shape = (batch, obs_dim, …)
        """

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[
        torch.Tensor,  # obs_next
        torch.Tensor,  # reward
        torch.Tensor,  # done mask (bool / 0-1 float)
        Dict[str, Any],
    ]:
        """
        Apply *batched* actions.

        Gradients **may** flow through `reward` and `obs_next` if the
        underlying physics supports it.
        """

class OracleBase(ABC):
    """
    Given the opponent's meta-strategy, train a (joint) best-response policy.
    """

    @abstractmethod
    def train_best_response(
        self,
        team_id: int,
        opponent_meta: "Population",
        *,
        steps: int,
        **kwargs,
    ) -> PolicyBase:
        """
        Returns
        -------
        PolicyBase  : the newly-trained best-response policy for `team_id`
        """


# --------------------------------------------------------------------------- #
# 4. Population container
# --------------------------------------------------------------------------- #
@dataclass
class Population:
    """
    A lightweight container for policies and their meta-weights.
    Weight semantics are left to the caller (uniform, Nash mix, etc.).

    Supports sampling without replacement for small pop sizes.
    """

    policies: List[PolicyBase] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)

    # ------------- basic operations ---------------- #
    def add(self, policy: PolicyBase, weight: float = 1.0) -> None:
        self.policies.append(policy)
        self.weights.append(float(weight))

    def __len__(self) -> int:  # len(pop)
        return len(self.policies)

    # ------------- utility helpers ---------------- #
    def normalize(self) -> None:
        """Normalize weights to sum to 1 in-place."""
        total = float(np.sum(self.weights))
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def as_numpy(self) -> np.ndarray:
        """Return weights as a 1-D numpy array (non-normalised)."""
        return np.asarray(self.weights, dtype=np.float32)

    def sample_indices(self, k: int) -> List[int]:
        """Sample k indices with probability proportional to current weights."""
        self.normalize()
        return list(np.random.choice(len(self), size=k, replace=False, p=self.as_numpy()))

    def sample(self, k: int) -> List[PolicyBase]:
        """Convenience wrapper around `sample_indices` returning policies."""
        return [self.policies[i] for i in self.sample_indices(k)]