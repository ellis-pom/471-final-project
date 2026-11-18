"""Scripted baseline defenders with the same interface as QLearningAgent.

They answer "is the learned policy better than something trivial?":
  * SleepAgent       - does nothing. Lower bound: the cost of no defence.
  * RandomAgent      - uniform over the 11 actions. What "no learning" looks like.
  * RestoreOnlyAgent - a simple human rule: Restore the most critical flagged
                       subnet (Op_Server > Enterprise > User), otherwise Sleep.
All three see exactly the same 18-state abstraction as the Q-agent, so any
difference is due to the policy, not to extra information.
"""
from typing import Optional

import numpy as np

from .abstraction import decode_state
from .actions import ACTION_LABELS, N_ACTIONS, SLEEP


class SleepAgent:
    name = "Sleep"

    def select_action(self, state: int, greedy: bool = True) -> int:
        return SLEEP


class RandomAgent:
    name = "Random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, greedy: bool = True) -> int:
        return int(self.rng.integers(N_ACTIONS))


class RestoreOnlyAgent:
    name = "RestoreOnly"
    RESTORE_OP = ACTION_LABELS.index("Restore-Operational")
    RESTORE_ENT = ACTION_LABELS.index("Restore-Enterprise")
    RESTORE_USER = ACTION_LABELS.index("Restore-User")

    def select_action(self, state: int, greedy: bool = True) -> int:
        user_bin, ent_bin, op_bin = decode_state(state)
        if op_bin:
            return self.RESTORE_OP
        if ent_bin:
            return self.RESTORE_ENT
        if user_bin:
            return self.RESTORE_USER
        return SLEEP
