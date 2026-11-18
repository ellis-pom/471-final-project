"""State abstraction: CybORG blue table -> one of 18 discrete states.

Everything here is a PURE function (no simulator, no randomness), so each rule
can be unit-tested with hand-built inputs (tests/test_abstraction.py).

The approved design compresses Blue's 13-host table into three features:

    user_flagged        flagged hosts among User0-4        -> 0 / 1 / 2+   (3 bins)
    enterprise_flagged  flagged hosts among Enterprise0-2  -> 0 / 1 / 2+   (3 bins)
    opserver_flagged    is Op_Server0 flagged?             -> no / yes     (2 bins)

    3 * 3 * 2 = 18 states, encoded as  index = user*6 + enterprise*2 + opserver.

Why so coarse? A tabular agent needs to visit every (state, action) pair many
times. 18 * 11 = 198 Q-values can be learned from a few hundred 30-step
episodes; the full table (13 hosts * 4 activity * 4 compromise levels) cannot.
The price is that the agent cannot tell WHICH user host is flagged -- the
action layer (actions.py) recovers that detail with a fixed rule.
"""
from typing import Dict, Tuple

# A host's status as Blue sees it: (Activity, Compromised), e.g. ("Scan", "No").
HostStatus = Dict[str, Tuple[str, str]]

# ---------------------------------------------------------------- host groups
# Verified against Scenario2.yaml (see DECISIONS.md, D2/D3).
USER_HOSTS = ["User0", "User1", "User2", "User3", "User4"]
# Defender is also in the Enterprise subnet, but it is Blue's own server and red
# never targets it, so we exclude it from counting and from targeting (D3).
ENTERPRISE_HOSTS = ["Enterprise0", "Enterprise1", "Enterprise2"]
OP_SERVER = "Op_Server0"
# Op_Server0 is listed first so it is the default target of Operational actions.
OPERATIONAL_HOSTS = [OP_SERVER, "Op_Host0", "Op_Host1", "Op_Host2"]

N_STATES = 18
BIN_LABELS = ["0", "1", "2+"]

# ---------------------------------------------------------------- the rules


def is_flagged(activity: str, compromised: str) -> bool:
    """THE single definition of "flagged" used everywhere in the project.

    A host is flagged if Blue's table shows ANY sign of red on it:
      * Activity != "None"  -> a Scan or Exploit was seen THIS step (this field
        only lasts one step, and green users also cause Scans: false positives), or
      * Compromised != "No" -> Blue believes red got in (User / Privileged), or
        Blue Removed red but cannot be sure it is clean ("Unknown").
    Only Restore sets Compromised back to "No", so only Restore fully un-flags a host.
    """
    return activity != "None" or compromised != "No"


def count_to_bin(count: int) -> int:
    """Map a flagged-host count to a bin: 0 -> 0, 1 -> 1, anything >= 2 -> 2 ("2+")."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return min(count, 2)


def count_flagged(host_status: HostStatus, hosts) -> int:
    """How many of `hosts` are flagged in `host_status`."""
    return sum(1 for h in hosts if is_flagged(*host_status[h]))


def encode_state(user_bin: int, ent_bin: int, op_bin: int) -> int:
    """(user 0-2, enterprise 0-2, opserver 0-1) -> single index 0..17 (mixed radix)."""
    if not (0 <= user_bin <= 2 and 0 <= ent_bin <= 2 and 0 <= op_bin <= 1):
        raise ValueError(f"bins out of range: {(user_bin, ent_bin, op_bin)}")
    return user_bin * 6 + ent_bin * 2 + op_bin


def decode_state(state: int) -> Tuple[int, int, int]:
    """Inverse of encode_state: 0..17 -> (user_bin, ent_bin, op_bin)."""
    if not 0 <= state < N_STATES:
        raise ValueError(f"state must be in 0..17, got {state}")
    return state // 6, (state % 6) // 2, state % 2


def state_label(state: int) -> str:
    """Human-readable label, e.g. 'User:2+ | Ent:1 | OpSrv:yes'."""
    u, e, o = decode_state(state)
    return f"User:{BIN_LABELS[u]} | Ent:{BIN_LABELS[e]} | OpSrv:{'yes' if o else 'no'}"


STATE_LABELS = [state_label(s) for s in range(N_STATES)]


def abstract_state(host_status: HostStatus) -> int:
    """Full Blue view (hostname -> (activity, compromised)) -> state index 0..17."""
    u = count_to_bin(count_flagged(host_status, USER_HOSTS))
    e = count_to_bin(count_flagged(host_status, ENTERPRISE_HOSTS))
    o = 1 if is_flagged(*host_status[OP_SERVER]) else 0
    return encode_state(u, e, o)


def table_to_host_status(table) -> HostStatus:
    """Convert BlueTableWrapper's PrettyTable to {hostname: (activity, compromised)}.

    Table columns are: Subnet, IP Address, Hostname, Activity, Compromised.
    """
    return {row[2]: (row[3], row[4]) for row in table.rows}
