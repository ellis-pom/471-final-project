"""AbstractDefenderEnv: the ONLY place our code touches CybORG.

    reset()      -> state (int 0..17)
    step(action) -> (state, reward, done, info)

The agent never sees CybORG objects. Swapping the state abstraction or the red
agent later (final project) only touches this file and abstraction.py.
"""
import inspect
import random
from typing import Dict, List, Tuple

import numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Shared.Actions import Analyse, Monitor, Remove, Restore, Sleep

from .abstraction import abstract_state, table_to_host_status
from .actions import ACTION_LABELS, resolve_action

RED_AGENTS = {"bline": B_lineAgent, "meander": RedMeanderAgent}
SCENARIO_PATH = inspect.getfile(CybORG)[: -len("CybORG.py")] + "Shared/Scenarios/Scenario2.yaml"

# Blue's only session in CAGE 2 is id 0 (verified with scripts/smoke_env.py).
BLUE_SESSION = 0


def seed_everything(seed: int) -> None:
    """CybORG draws all randomness (green agent, Meander, IP layout, action
    outcomes) from Python's global `random`; we also seed NumPy for completeness."""
    random.seed(seed)
    np.random.seed(seed)


def make_cyborg_action(verb: str, hostname):
    """(verb, hostname) from actions.resolve_action -> real CybORG action object."""
    if verb == "Sleep":
        return Sleep()
    if verb == "Monitor":
        return Monitor(session=BLUE_SESSION, agent="Blue")
    cls = {"Analyse": Analyse, "Remove": Remove, "Restore": Restore}[verb]
    return cls(session=BLUE_SESSION, agent="Blue", hostname=hostname)


class AbstractDefenderEnv:
    """Fixed-length CAGE 2 episodes seen through our 18-state / 11-action abstraction.

    Args:
        red: "bline" (B_lineAgent: beelines to Op_Server0) or "meander"
             (RedMeanderAgent: spreads through every host first).
        max_steps: episode length. CybORG never ends an episode by itself, so we
             cut it at max_steps (a time limit, not a real terminal state; D8).
        seed: if given, seeds `random`/NumPy ONCE here, so a whole sequence of
             episodes is reproducible from this one number.
    """

    def __init__(self, red: str = "bline", max_steps: int = 30, seed=None):
        if red not in RED_AGENTS:
            raise ValueError(f"red must be one of {list(RED_AGENTS)}")
        if seed is not None:
            seed_everything(seed)
        self.red = red
        self.max_steps = max_steps
        self.cyborg = CybORG(SCENARIO_PATH, "sim", agents={"Red": RED_AGENTS[red]})
        # output_mode="table": we parse the human-readable blue table, the same
        # thing we printed in the smoke test, rather than CybORG's opaque vector.
        self.env = BlueTableWrapper(env=self.cyborg, output_mode="table")
        self.t = 0
        self.host_status = None

    def reset(self) -> int:
        result = self.env.reset(agent="Blue")
        self.t = 0
        self.host_status = table_to_host_status(result.observation)
        return abstract_state(self.host_status)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        # The targeting rule looks at the CURRENT table (what Blue sees now).
        verb, host = resolve_action(action, self.host_status)
        result = self.env.step(agent="Blue", action=make_cyborg_action(verb, host))
        self.t += 1
        self.host_status = table_to_host_status(result.observation)
        state = abstract_state(self.host_status)
        # Default CAGE 2 reward, passed through unchanged (already includes
        # Restore's -1 cost).
        reward = float(result.reward)
        done = self.t >= self.max_steps
        info = {
            "t": self.t,
            "action_label": ACTION_LABELS[action],
            "verb": verb,
            "target_host": host,
            "blue_action": str(self.cyborg.get_last_action("Blue")),
            "red_action": str(self.cyborg.get_last_action("Red")),
            "host_status": dict(self.host_status),
        }
        return state, reward, done, info

    def red_footprint(self) -> Dict[str, List[str]]:
        """GROUND TRUTH (Blue cannot see this): where red actually has sessions.

        Used only for evaluation metrics and traces, never by the agent.
        Returns {"any": hosts with any red session, "privileged": hosts with a
        root/SYSTEM red session -- the ones the reward penalises}.
        """
        true_state = self.cyborg.get_agent_state("True")
        any_hosts, priv_hosts = [], []
        for host, info in true_state.items():
            if host == "success":
                continue
            red_sessions = [s for s in info.get("Sessions", []) if s.get("Agent") == "Red"]
            if red_sessions:
                any_hosts.append(host)
            if any(s.get("Username") in ("root", "SYSTEM") for s in red_sessions):
                priv_hosts.append(host)
        return {"any": sorted(any_hosts), "privileged": sorted(priv_hosts)}
