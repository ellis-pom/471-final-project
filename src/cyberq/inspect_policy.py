"""Explain a trained Q-table: policy table, Q-table CSV and state coverage.

    python -m cyberq.inspect_policy --qtable results/run_bline_s0/qtable.npy

For each of the 18 states it prints the greedy action, its Q-value, the runner-up
action, the gap between them, and how many training updates that state got.
Writes policy.csv and qtable.csv (open them in Excel or pandas).

How to read it:
  * visits == 0  -> state NEVER seen in training; its "policy" is just the
                    zero-initialised default (Sleep) and means nothing.
  * small gap    -> the agent barely prefers its choice; low confidence.
  * an action with 0 visits in a visited state keeps Q = 0, which (because all
    rewards are negative) can look "best" without any evidence. Column
    best_visits shows how much evidence backs the chosen action.
"""
import argparse
import csv
import os

import numpy as np

from .abstraction import STATE_LABELS
from .actions import ACTION_LABELS
from .q_agent import QLearningAgent


def policy_rows(agent: QLearningAgent):
    rows = []
    for s in range(agent.n_states):
        q = agent.q[s]
        order = np.argsort(-q, kind="stable")   # stable: ties keep lowest index first, like argmax
        best, second = int(order[0]), int(order[1])
        rows.append({
            "state": s, "label": STATE_LABELS[s],
            "best_action": ACTION_LABELS[best], "best_q": float(q[best]),
            "runner_up": ACTION_LABELS[second], "runner_up_q": float(q[second]),
            "gap": float(q[best] - q[second]),
            "state_visits": int(agent.visits[s].sum()),
            "best_visits": int(agent.visits[s, best]),
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qtable", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    agent = QLearningAgent.load(args.qtable)
    out_dir = args.out or (os.path.dirname(args.qtable) if args.qtable.endswith(".npy") else args.qtable)
    rows = policy_rows(agent)
    total_visits = agent.visits.sum()

    print(f"Policy for Q-table trained vs red={agent.meta.get('red_agent')} "
          f"for {agent.episodes_trained} episodes\n")
    print(f"{'s':>2}  {'state':<28} {'greedy action':<20} {'Q':>7}  {'runner-up':<20} {'Q':>7} "
          f"{'gap':>6} {'visits':>7} {'%':>5}")
    for r in rows:
        note = "  <- NEVER VISITED: default, do not interpret" if r["state_visits"] == 0 else (
            "  <- chosen action never tried here" if r["best_visits"] == 0 else "")
        pct = 100.0 * r["state_visits"] / total_visits if total_visits else 0.0
        print(f"{r['state']:>2}  {r['label']:<28} {r['best_action']:<20} {r['best_q']:>7.2f}  "
              f"{r['runner_up']:<20} {r['runner_up_q']:>7.2f} {r['gap']:>6.2f} {r['state_visits']:>7} {pct:>5.1f}{note}")

    visited = sum(1 for r in rows if r["state_visits"] > 0)
    print(f"\nState coverage: {visited}/18 states visited during training "
          f"({total_visits} updates total).")
    print(f"(state, action) coverage: {int(np.count_nonzero(agent.visits))}/{agent.visits.size} pairs updated at least once.")

    with open(os.path.join(out_dir, "policy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "qtable.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "label"] + ACTION_LABELS)
        for s in range(agent.n_states):
            w.writerow([s, STATE_LABELS[s]] + [round(float(v), 4) for v in agent.q[s]])
    with open(os.path.join(out_dir, "visits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "label"] + ACTION_LABELS + ["total"])
        for s in range(agent.n_states):
            w.writerow([s, STATE_LABELS[s]] + [int(v) for v in agent.visits[s]] + [int(agent.visits[s].sum())])
    print(f"Wrote policy.csv, qtable.csv and visits.csv to {out_dir}/")


if __name__ == "__main__":
    main()
