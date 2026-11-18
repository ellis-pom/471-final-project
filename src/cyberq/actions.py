"""Action table (11 abstract actions) and the "most suspicious host" rule.

The agent picks a SUBNET-level action (e.g. "Restore Enterprise"); this module
turns it into a concrete (verb, hostname) pair. Pure functions only, so the
targeting rule is unit-tested without the simulator (tests/test_actions.py).
env_wrapper.py then builds the real CybORG action object from (verb, hostname).
"""
from typing import List, Optional, Tuple

from .abstraction import ENTERPRISE_HOSTS, OPERATIONAL_HOSTS, USER_HOSTS, HostStatus

SUBNET_HOSTS = {
    "User": USER_HOSTS,
    "Enterprise": ENTERPRISE_HOSTS,
    "Operational": OPERATIONAL_HOSTS,
}

# (verb, subnet). subnet is None for global actions. Index = action id 0..10.
# Note: CybORG already runs Monitor automatically after every step (DECISIONS D4),
# so "Monitor" behaves like "Sleep". We keep it to match the approved design.
ACTIONS: List[Tuple[str, Optional[str]]] = [("Sleep", None), ("Monitor", None)]
for _verb in ("Analyse", "Remove", "Restore"):
    for _subnet in ("User", "Enterprise", "Operational"):
        ACTIONS.append((_verb, _subnet))

N_ACTIONS = len(ACTIONS)  # 11
ACTION_LABELS = [verb if subnet is None else f"{verb}-{subnet}" for verb, subnet in ACTIONS]
SLEEP = ACTION_LABELS.index("Sleep")

# Higher number = more suspicious.
COMPROMISED_RANK = {"No": 0, "Unknown": 1, "User": 2, "Privileged": 3}
ACTIVITY_RANK = {"None": 0, "Scan": 1, "Exploit": 2}


def suspicion_key(activity: str, compromised: str) -> Tuple[int, int]:
    """Sort key for suspicion: compromise level first, then this step's activity.

    Compromise comes first because it is persistent evidence that red holds the
    host; Activity is a one-step signal that may be a green false positive.
    """
    return COMPROMISED_RANK[compromised], ACTIVITY_RANK[activity]


def most_suspicious_host(host_status: HostStatus, subnet: str) -> str:
    """Pick the host to act on inside `subnet`. Deterministic.

    Rank by (Compromised: Privileged > User > Unknown > No, then
    Activity: Exploit > Scan > None). Ties go to the host listed FIRST in
    SUBNET_HOSTS (fixed order). If nothing is flagged, every host ties at (0, 0)
    so we fall back to the first host: User0 / Enterprise0 / Op_Server0.
    """
    hosts = SUBNET_HOSTS[subnet]
    best = hosts[0]
    for h in hosts[1:]:
        # strict ">" means an earlier host wins any tie
        if suspicion_key(*host_status[h]) > suspicion_key(*host_status[best]):
            best = h
    return best


def resolve_action(action: int, host_status: HostStatus) -> Tuple[str, Optional[str]]:
    """Abstract action id -> (verb, hostname or None)."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {action}")
    verb, subnet = ACTIONS[action]
    if subnet is None:
        return verb, None
    return verb, most_suspicious_host(host_status, subnet)
