"""Train the Q-learning defender.

    python -m cyberq.train --episodes 500 --steps 30 --red bline --seed 0 --out results/run_bline_s0

Outputs in --out:  qtable.npy, visits.npy, meta.json, train_log.csv,
                   (optional) reward_curve.png
"""
import argparse
import csv
import os
import time

import numpy as np

from . import config
from .abstraction import N_STATES, STATE_LABELS
from .actions import ACTION_LABELS, N_ACTIONS
from .env_wrapper import AbstractDefenderEnv
from .q_agent import QLearningAgent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    p.add_argument("--steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--red", default=config.RED, choices=["bline", "meander"])
    p.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    p.add_argument("--alpha", type=float, default=config.ALPHA)
    p.add_argument("--gamma", type=float, default=config.GAMMA)
    p.add_argument("--eps-start", type=float, default=config.EPS_START)
    p.add_argument("--eps-end", type=float, default=config.EPS_END)
    p.add_argument("--eps-decay-episodes", type=int, default=None,
                   help=f"default: {config.EPS_DECAY_FRACTION:.0%} of --episodes")
    p.add_argument("--out", default="results/run")
    p.add_argument("--log-every", type=int, default=10, help="print every N episodes (CSV has all)")
    p.add_argument("--plot", action="store_true", help="save reward_curve.png (needs matplotlib)")
    return p.parse_args(argv)


def train(args) -> QLearningAgent:
    decay = args.eps_decay_episodes or max(1, int(args.episodes * config.EPS_DECAY_FRACTION))
    env = AbstractDefenderEnv(red=args.red, max_steps=args.steps, seed=args.seed)
    agent = QLearningAgent(N_STATES, N_ACTIONS, alpha=args.alpha, gamma=args.gamma,
                           eps_start=args.eps_start, eps_end=args.eps_end,
                           eps_decay_episodes=decay, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.csv")
    print(f"Training {args.episodes} episodes x {args.steps} steps vs red={args.red}, seed={args.seed}")
    print(f"alpha={args.alpha} gamma={args.gamma} eps {args.eps_start}->{args.eps_end} over {decay} eps")
    print(f"{'episode':>7} {'return':>8} {'epsilon':>7} {'states_seen':>11} {'nonzero_Q':>9} {'mean|TD|':>8}")

    rewards = []
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["episode", "return", "epsilon", "states_seen", "nonzero_q", "mean_abs_td"])
        for ep in range(1, args.episodes + 1):
            epsilon_used = agent.epsilon          # epsilon in effect THIS episode
            state = env.reset()
            total, td_sum, done = 0.0, 0.0, False
            # ---------------- the Q-learning loop ----------------
            while not done:
                action = agent.select_action(state)                 # epsilon-greedy
                next_state, reward, done, _ = env.step(action)      # act in CybORG
                # terminal=False even at the time limit: step 30 is not a real
                # end state, the attack would continue (DECISIONS D8).
                td_sum += abs(agent.update(state, action, reward, next_state, terminal=False))
                total += reward
                state = next_state
            agent.end_episode()                                     # decay epsilon
            # ------------------------------------------------------
            rewards.append(total)
            states_seen = int((agent.visits.sum(axis=1) > 0).sum())
            nonzero = int(np.count_nonzero(agent.q))
            mean_td = td_sum / args.steps
            log.writerow([ep, round(total, 2), round(epsilon_used, 4), states_seen, nonzero, round(mean_td, 4)])
            if ep % args.log_every == 0 or ep == 1 or ep == args.episodes:
                print(f"{ep:>7} {total:>8.1f} {epsilon_used:>7.3f} {states_seen:>8}/18 {nonzero:>6}/{agent.q.size} {mean_td:>8.3f}")

    agent.save(args.out, extra_meta={
        "red_agent": args.red, "max_steps": args.steps, "train_seed": args.seed,
        "state_labels": STATE_LABELS, "action_labels": ACTION_LABELS,
    })
    last = rewards[-50:]
    print(f"\nDone in {time.time() - t0:.1f}s. Last {len(last)} episodes: mean return "
          f"{np.mean(last):.1f} +/- {np.std(last):.1f} (these still include epsilon={args.eps_end} exploration)")
    print(f"Saved Q-table, visit counts and metadata to {args.out}/")
    if args.plot:
        plot_rewards(rewards, os.path.join(args.out, "reward_curve.png"))
    return agent


def plot_rewards(rewards, path, window=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.asarray(rewards)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(r) + 1), r, alpha=0.35, label="episode return")
    if len(r) >= window:
        ma = np.convolve(r, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(r) + 1), ma, label=f"{window}-episode moving average")
    ax.set_xlabel("episode")
    ax.set_ylabel("return (sum of rewards)")
    ax.set_title("Training return per episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    train(parse_args())
