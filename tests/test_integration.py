"""Integration: wrapper + real CybORG simulator + agents."""
import numpy as np
import pytest

from cyberq.abstraction import OPERATIONAL_HOSTS
from cyberq.actions import ACTION_LABELS, N_ACTIONS
from cyberq.env_wrapper import AbstractDefenderEnv
from cyberq.evaluate import GreedyQ, run_episode
from cyberq.q_agent import QLearningAgent

pytestmark = pytest.mark.slow


def test_wrapper_runs_real_episodes_with_valid_states():
    """Correctness property: two short episodes against the real simulator run
    without crashing, states stay in 0..17, every action 0..10 is accepted, an
    episode ends exactly at max_steps, and rewards are never positive (CAGE 2 Blue)."""
    env = AbstractDefenderEnv(red="bline", max_steps=12, seed=123)
    for ep in range(2):
        s = env.reset()
        assert 0 <= s < 18
        for t in range(12):
            s, r, done, info = env.step(t % N_ACTIONS)
            assert 0 <= s < 18
            assert r <= 0
            assert done == (t == 11)
            if info["action_label"] == "Restore-Operational":
                assert info["target_host"] in OPERATIONAL_HOSTS


def test_same_seed_gives_same_episode():
    """Correctness property: reproducibility - two environments built with the
    same seed and fed the same actions produce identical rewards and states."""
    def rollout():
        env = AbstractDefenderEnv(red="meander", max_steps=15, seed=7)
        env.reset()
        return [env.step(a % N_ACTIONS)[:2] for a in range(15)]
    assert rollout() == rollout()


def test_evaluation_does_not_modify_q_table():
    """Correctness property: running greedy evaluation episodes (epsilon = 0)
    leaves the Q-table and visit counts unchanged - evaluation never learns."""
    agent = QLearningAgent(18, N_ACTIONS, seed=0)
    agent.q = np.random.default_rng(0).normal(size=agent.q.shape)
    q_before, v_before = agent.q.copy(), agent.visits.copy()
    env = AbstractDefenderEnv(red="bline", max_steps=10, seed=5)
    for _ in range(2):
        run_episode(env, GreedyQ(agent))
    assert np.array_equal(agent.q, q_before) and np.array_equal(agent.visits, v_before)
