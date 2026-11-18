"""Core Q-learning math: the TD update, greedy action choice, and epsilon decay."""
import numpy as np
import pytest

from cyberq.q_agent import QLearningAgent


def make_agent(**kw):
    return QLearningAgent(n_states=3, n_actions=2, seed=0, **kw)


def test_single_update_matches_hand_calculation():
    """Correctness property: one call to update() applies exactly
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).

    Hand calculation: Q(0,1)=2.0, max Q(1,.)=max(5,-3)=5, r=-1, alpha=0.1, gamma=0.9
      target = -1 + 0.9*5 = 3.5;  TD error = 3.5 - 2.0 = 1.5;  new Q = 2.0 + 0.1*1.5 = 2.15
    """
    agent = make_agent(alpha=0.1, gamma=0.9)
    agent.q[0, 1] = 2.0
    agent.q[1] = [5.0, -3.0]
    td = agent.update(s=0, a=1, r=-1.0, s_next=1, terminal=False)
    assert td == pytest.approx(1.5)
    assert agent.q[0, 1] == pytest.approx(2.15)
    # nothing else in the table moved, and the visit was counted
    assert agent.q[0, 0] == 0.0 and list(agent.q[1]) == [5.0, -3.0]
    assert agent.visits[0, 1] == 1 and agent.visits.sum() == 1


def test_terminal_update_ignores_bootstrap():
    """Correctness property: at a terminal transition the target is just r;
    the value of s' is ignored (there is no future after the end)."""
    agent = make_agent(alpha=0.5, gamma=0.9)
    agent.q[0, 0] = 1.0
    agent.q[2] = [100.0, 100.0]          # would dominate if it were (wrongly) used
    agent.update(s=0, a=0, r=-2.0, s_next=2, terminal=True)
    assert agent.q[0, 0] == pytest.approx(1.0 + 0.5 * (-2.0 - 1.0))   # = -0.5


def test_greedy_mode_is_argmax_and_never_explores():
    """Correctness property: with greedy=True the agent ignores epsilon (even
    epsilon = 1.0) and always returns argmax_a Q(s,a); ties go to the lowest index."""
    agent = make_agent(eps_start=1.0)
    assert agent.epsilon == 1.0
    agent.q[0] = [0.0, 3.0]
    agent.q[1] = [-1.0, -1.0]             # tie
    assert all(agent.select_action(0, greedy=True) == 1 for _ in range(200))
    assert all(agent.select_action(1, greedy=True) == 0 for _ in range(200))


def test_epsilon_greedy_explores_when_epsilon_is_one():
    """Correctness property: in training mode with epsilon = 1 the choice is
    uniformly random, so a non-greedy action is taken roughly half the time."""
    agent = make_agent(eps_start=1.0)
    agent.q[0] = [0.0, 3.0]
    picks = [agent.select_action(0) for _ in range(2000)]
    assert 0.4 < picks.count(0) / len(picks) < 0.6


def test_epsilon_decays_linearly_to_floor():
    """Correctness property: epsilon falls linearly from eps_start to eps_end
    over eps_decay_episodes and then stays at the floor."""
    agent = make_agent(eps_start=1.0, eps_end=0.1, eps_decay_episodes=10)
    eps = []
    for _ in range(15):
        agent.end_episode()
        eps.append(agent.epsilon)
    assert eps[4] == pytest.approx(1.0 - 0.9 * 5 / 10)    # after 5 episodes: 0.55
    assert eps[9] == pytest.approx(0.1)
    assert eps[14] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(eps, eps[1:]))       # never increases
