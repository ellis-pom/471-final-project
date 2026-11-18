"""Smoke test: reset CybORG CAGE 2, take a few Sleep steps, print the raw blue view.

Why this exists: before we write any abstraction code we want to SEE the real
observation format instead of trusting our assumptions about it.

Run:  python scripts/smoke_env.py [--red bline|meander] [--steps 8] [--seed 0]
"""
import argparse
import inspect
import random

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Shared.Actions import Sleep

RED = {"bline": B_lineAgent, "meander": RedMeanderAgent}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--red", default="bline", choices=RED)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Scenario2.yaml ships inside the CybORG package; locate it relative to CybORG.py.
    path = inspect.getfile(CybORG)[:-10] + "/Shared/Scenarios/Scenario2.yaml"
    cyborg = CybORG(path, "sim", agents={"Red": RED[args.red]})
    env = BlueTableWrapper(env=cyborg, output_mode="table")

    res = env.reset(agent="Blue")
    print("=== reset: blue table ===")
    print(res.observation)
    space = res.action_space
    print("blue action classes:", [a.__name__ for a in space["action"]])
    print("blue session ids:", list(space["session"]))
    print("blue hostnames:", sorted(space["hostname"]))

    for t in range(1, args.steps + 1):
        res = env.step(agent="Blue", action=Sleep())
        red_action = cyborg.get_last_action("Red")
        print(f"\n=== step {t} | blue reward {res.reward} | done {res.done} | red did: {red_action} ===")
        print(res.observation)


if __name__ == "__main__":
    main()
