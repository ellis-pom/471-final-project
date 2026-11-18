"""Tabular Q-learning agent (off-policy TD control, epsilon-greedy behaviour).

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The agent knows nothing about CybORG: it sees integers (states 0..n_states-1,
actions 0..n_actions-1) and float rewards. That is why the same class can be
tested on a tiny toy MDP (tests/test_toy_mdp.py) where we know the answer.
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np


class QLearningAgent:
    """Args:
        n_states, n_actions: size of the Q-table (18 x 11 for our design).
        alpha: learning rate - how far each update moves Q toward the TD target.
        gamma: discount - how much future reward counts (0.95 ~ 20-step horizon).
        eps_start, eps_end: exploration probability at the start / floor.
        eps_decay_episodes: episodes over which epsilon falls LINEARLY from
            eps_start to eps_end; it then stays at eps_end.
        seed: seeds the agent's own RNG (exploration), separate from the simulator.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_episodes: int = 400,
                 seed: Optional[int] = None):
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.epsilon = eps_start
        self.episodes_trained = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Zero init. CAGE 2 rewards are <= 0, so an untried action (Q=0) looks
        # BETTER than any tried one: zero init acts as optimistic init and pushes
        # the agent to try every action in a state at least once (D9).
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        # How many times each (s, a) was UPDATED during training. Tells us which
        # parts of the table are actually learned vs. still the zero default.
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)

    # ------------------------------------------------------------ acting
    def greedy_action(self, state: int) -> int:
        """argmax_a Q(state, a). Ties -> LOWEST action index (np.argmax rule).

        Deterministic tie-breaking keeps evaluation reproducible and easy to
        explain; the catch is that an unvisited state (all zeros) always maps to
        action 0 (Sleep). inspect_policy flags those states.
        """
        return int(np.argmax(self.q[state]))

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy during training; pure greedy (epsilon = 0) if greedy=True."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))  # explore: uniform random
        return self.greedy_action(state)                   # exploit

    # ------------------------------------------------------------ learning
    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool) -> float:
        """One Q-learning (TD(0)) update. Returns the TD error.

        terminal=True means s_next is a true end state with no future, so the
        target is just r. Our CAGE 2 episodes end only because of a time limit,
        so train.py passes terminal=False and keeps bootstrapping (D8).
        """
        best_next = 0.0 if terminal else float(np.max(self.q[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        self.visits[s, a] += 1
        return float(td_error)

    def end_episode(self) -> None:
        """Call once after every training episode: count it and decay epsilon linearly."""
        self.episodes_trained += 1
        frac = min(1.0, self.episodes_trained / self.eps_decay_episodes)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    # ------------------------------------------------------------ persistence
    def hyperparams(self) -> dict:
        return {
            "n_states": self.n_states, "n_actions": self.n_actions,
            "alpha": self.alpha, "gamma": self.gamma,
            "eps_start": self.eps_start, "eps_end": self.eps_end,
            "eps_decay_episodes": self.eps_decay_episodes, "seed": self.seed,
        }

    def save(self, out_dir: str, extra_meta: Optional[dict] = None) -> None:
        """Write qtable.npy, visits.npy and meta.json (hyperparameters + context)."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "qtable.npy"), self.q)
        np.save(os.path.join(out_dir, "visits.npy"), self.visits)
        meta = {
            "hyperparams": self.hyperparams(),
            "episodes_trained": self.episodes_trained,
            "epsilon_final": self.epsilon,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        meta.update(extra_meta or {})
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load from a run directory, or from the path of its qtable.npy."""
        out_dir = os.path.dirname(path) if path.endswith(".npy") else path
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        agent = cls(**meta["hyperparams"])
        agent.q = np.load(os.path.join(out_dir, "qtable.npy"))
        visits_path = os.path.join(out_dir, "visits.npy")
        if os.path.exists(visits_path):
            agent.visits = np.load(visits_path)
        agent.episodes_trained = meta["episodes_trained"]
        agent.epsilon = meta["epsilon_final"]
        agent.meta = meta
        return agent
