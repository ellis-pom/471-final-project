"""Learning sanity check: the SAME QLearningAgent class solves a tiny MDP with a known answer.

Corridor:   [0] - [1] - [2] - [3=goal, terminal]
Actions:    0 = left (wall at state 0), 1 = right
Reward:     -1 on every step (negative like CAGE 2), episode ends at the goal.
Optimal:    always go right. With gamma = 0.9 the optimal values are
            Q*(2,R) = -1,  Q*(1,R) = -1 - 0.9 = -1.9,  Q*(0,R) = -1.9*0.9 - 1 = -2.71
"""
import numpy as np
import pytest

from cyberq.q_agent import QLearningAgent

GOAL = 3


def corridor_step(s, a):
    s_next = max(0, s - 1) if a == 0 else s + 1
    return s_next, -1.0, s_next == GOAL


def train_corridor(episodes=300, seed=0):
    agent = QLearningAgent(n_states=4, n_actions=2, alpha=0.5, gamma=0.9,
                           eps_start=1.0, eps_end=0.05, eps_decay_episodes=200, seed=seed)
    for _ in range(episodes):
        s, done, t = 0, False, 0
        while not done and t < 50:
            a = agent.select_action(s)
            s_next, r, done = corridor_step(s, a)
            agent.update(s, a, r, s_next, terminal=done)
            s, t = s_next, t + 1
        agent.end_episode()
    return agent


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_learns_known_optimal_policy(seed):
    """Correctness property: after training, the greedy policy equals the known
    optimum (right in every non-goal state) for several seeds, and the learned
    Q-values match the hand-computed Q* within 0.05. This is the strongest
    evidence that select_action + update + epsilon decay form a working
    Q-learning loop, independent of CybORG."""
    agent = train_corridor(seed=seed)
    assert [agent.greedy_action(s) for s in range(3)] == [1, 1, 1]
    assert agent.q[0:3, 1] == pytest.approx([-2.71, -1.9, -1.0], abs=0.05)
