"""Evaluate the learned greedy policy (epsilon = 0, NO Q-table updates) against baselines.

    python -m cyberq.evaluate --qtable results/run_bline_s0/qtable.npy --episodes 50 --red bline --seed 100 [--trace]

Every agent gets a fresh environment seeded with the SAME --seed, so all of them
face the same random draws at the start (identical conditions). Use a seed that
was not used in training. Writes eval_results.csv (one row per episode) and
action_distribution.csv next to the Q-table (or to --out).
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np

from . import config
from .abstraction import STATE_LABELS
from .actions import ACTION_LABELS, N_ACTIONS
from .baselines import RandomAgent, RestoreOnlyAgent, SleepAgent
from .env_wrapper import AbstractDefenderEnv
from .q_agent import QLearningAgent


class GreedyQ:
    """Thin adapter: always call the Q-agent with greedy=True (epsilon = 0).
    run_episode never calls update(), so the Q-table cannot change during evaluation."""
    name = "Q-agent"

    def __init__(self, agent: QLearningAgent):
        self.agent = agent

    def select_action(self, state, greedy=True):
        return self.agent.select_action(state, greedy=True)


def run_episode(env: AbstractDefenderEnv, policy, trace: bool = False) -> dict:
    """Play one episode with no learning. Returns per-episode metrics."""
    state = env.reset()
    total, done = 0.0, False
    counts = Counter()
    if trace:
        print(f"\n--- Episode trace: {policy.name} vs red={env.red} ---")
        print(f"{'t':>2}  {'state seen by Blue':<28} {'action':<20} {'target':<11} {'reward':>6}  red did")
    while not done:
        action = policy.select_action(state)
        next_state, reward, done, info = env.step(action)
        counts[action] += 1
        total += reward
        if trace:
            print(f"{info['t']:>2}  {STATE_LABELS[state]:<28} {info['action_label']:<20} "
                  f"{str(info['target_host'] or '-'):<11} {reward:>6.1f}  {info['red_action']}")
        state = next_state
    foot = env.red_footprint()
    if trace:
        print(f"End: return {total:.1f}. Red truly holds: {foot['any']} "
              f"(privileged on {foot['privileged']})")
    return {
        "return": total,
        "action_counts": counts,
        "restores": sum(c for a, c in counts.items() if ACTION_LABELS[a].startswith("Restore")),
        "red_hosts_end": len(foot["any"]),
        "red_privileged_end": len(foot["privileged"]),
        "opserver_lost": int("Op_Server0" in foot["privileged"]),
    }


def evaluate(args):
    q_agent = QLearningAgent.load(args.qtable)
    meta = q_agent.meta
    if meta.get("train_seed") == args.seed:
        print(f"WARNING: eval seed {args.seed} equals the training seed")
    out_dir = args.out or (os.path.dirname(args.qtable) if args.qtable.endswith(".npy") else args.qtable)
    os.makedirs(out_dir, exist_ok=True)

    policies = [GreedyQ(q_agent), RandomAgent(seed=args.seed), SleepAgent(), RestoreOnlyAgent()]
    print(f"Evaluating {args.episodes} episodes x {args.steps} steps vs red={args.red}, seed={args.seed} "
          f"(Q-table trained vs red={meta.get('red_agent')} for {meta.get('episodes_trained')} episodes)")

    rows, summary = [], {}
    q_before = q_agent.q.copy()
    for pol in policies:
        env = AbstractDefenderEnv(red=args.red, max_steps=args.steps, seed=args.seed)
        results = [run_episode(env, pol) for _ in range(args.episodes)]
        summary[pol.name] = results
        for i, r in enumerate(results):
            rows.append([pol.name, i, round(r["return"], 2), r["restores"], r["red_hosts_end"],
                         r["red_privileged_end"], r["opserver_lost"]])
    assert np.array_equal(q_before, q_agent.q), "Q-table changed during evaluation!"

    # ---------------------------------------------------------- results table
    print(f"\n{'agent':<12} {'mean return':>12} {'std':>6} {'min':>7} {'max':>7} "
          f"{'restores/ep':>11} {'red hosts@end':>13} {'red priv@end':>12} {'OpSrv lost':>10}")
    for name, res in summary.items():
        ret = np.array([r["return"] for r in res])
        print(f"{name:<12} {ret.mean():>12.1f} {ret.std():>6.1f} {ret.min():>7.1f} {ret.max():>7.1f} "
              f"{np.mean([r['restores'] for r in res]):>11.1f} "
              f"{np.mean([r['red_hosts_end'] for r in res]):>13.1f} "
              f"{np.mean([r['red_privileged_end'] for r in res]):>12.1f} "
              f"{np.mean([r['opserver_lost'] for r in res]):>9.0%}")
    print("(red hosts / priv / OpSrv lost = simulator GROUND TRUTH at the final step; "
          "User0 is red's foothold and is always counted)")

    # ---------------------------------------------------------- action distribution
    print("\nAction distribution (% of all steps):")
    print(f"{'action':<20}" + "".join(f"{n:>12}" for n in summary))
    dist = {}
    for name, res in summary.items():
        total = Counter()
        for r in res:
            total.update(r["action_counts"])
        n = sum(total.values())
        dist[name] = [100.0 * total[a] / n for a in range(N_ACTIONS)]
    for a in range(N_ACTIONS):
        print(f"{ACTION_LABELS[a]:<20}" + "".join(f"{dist[n][a]:>11.1f}%" for n in summary))

    with open(os.path.join(out_dir, "eval_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "episode", "return", "restores", "red_hosts_end",
                    "red_privileged_end", "opserver_lost"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "action_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["action"] + list(summary))
        for a in range(N_ACTIONS):
            w.writerow([ACTION_LABELS[a]] + [round(dist[n][a], 2) for n in summary])
    print(f"\nWrote eval_results.csv and action_distribution.csv to {out_dir}/")

    if args.trace:
        env = AbstractDefenderEnv(red=args.red, max_steps=args.steps, seed=args.seed)
        run_episode(env, GreedyQ(q_agent), trace=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qtable", required=True, help="path to qtable.npy or its run directory")
    p.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
    p.add_argument("--steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--red", default=config.RED, choices=["bline", "meander"])
    p.add_argument("--seed", type=int, default=config.EVAL_SEED)
    p.add_argument("--out", default=None, help="output dir (default: the Q-table's directory)")
    p.add_argument("--trace", action="store_true", help="also print one Q-agent episode step by step")
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
