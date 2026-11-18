"""Persistence: a saved agent loads back identically."""
import json

import numpy as np

from cyberq.q_agent import QLearningAgent


def test_save_load_round_trip(tmp_path):
    """Correctness property: save() then load() reproduces the Q-table, visit
    counts, hyperparameters, epsilon and episode count exactly, and the JSON
    sidecar carries the extra run metadata."""
    a = QLearningAgent(18, 11, alpha=0.2, gamma=0.9, eps_start=1.0, eps_end=0.05,
                       eps_decay_episodes=7, seed=3)
    rng = np.random.default_rng(0)
    a.q = rng.normal(size=(18, 11))
    a.visits = rng.integers(0, 50, size=(18, 11))
    for _ in range(4):
        a.end_episode()
    a.save(str(tmp_path), extra_meta={"red_agent": "bline", "train_seed": 3})

    b = QLearningAgent.load(str(tmp_path / "qtable.npy"))
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.visits, b.visits)
    assert a.hyperparams() == b.hyperparams()
    assert b.epsilon == a.epsilon and b.episodes_trained == 4
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["red_agent"] == "bline" and meta["train_seed"] == 3
