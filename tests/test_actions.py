"""Action table and the most-suspicious-host targeting rule."""
from cyberq.abstraction import ENTERPRISE_HOSTS, OPERATIONAL_HOSTS, USER_HOSTS
from cyberq.actions import ACTION_LABELS, N_ACTIONS, most_suspicious_host, resolve_action

ALL_HOSTS = USER_HOSTS + ENTERPRISE_HOSTS + OPERATIONAL_HOSTS + ["Defender"]


def clean():
    return {h: ("None", "No") for h in ALL_HOSTS}


def test_action_table_matches_design():
    """Correctness property: there are exactly 11 actions in the approved order
    (Sleep, Monitor, then {Analyse, Remove, Restore} x {User, Enterprise, Operational})."""
    assert N_ACTIONS == 11
    assert ACTION_LABELS[:2] == ["Sleep", "Monitor"]
    assert ACTION_LABELS[2:5] == ["Analyse-User", "Analyse-Enterprise", "Analyse-Operational"]
    assert ACTION_LABELS[-1] == "Restore-Operational"


def test_nothing_flagged_uses_fixed_default():
    """Correctness property: with no flagged host the rule targets the subnet's
    fixed default (first host): User0, Enterprise0, Op_Server0."""
    v = clean()
    assert most_suspicious_host(v, "User") == "User0"
    assert most_suspicious_host(v, "Enterprise") == "Enterprise0"
    assert most_suspicious_host(v, "Operational") == "Op_Server0"


def test_compromise_outranks_activity():
    """Correctness property: Compromised level is ranked before Activity, so a
    host red already holds (User) beats one merely being scanned or exploited."""
    v = clean()
    v["User1"] = ("Exploit", "No")
    v["User3"] = ("None", "User")
    assert most_suspicious_host(v, "User") == "User3"
    v["User4"] = ("None", "Privileged")
    assert most_suspicious_host(v, "User") == "User4"


def test_activity_breaks_compromise_ties_then_host_order():
    """Correctness property: equal compromise -> higher Activity wins
    (Exploit > Scan); a full tie goes to the host listed first."""
    v = clean()
    v["Enterprise2"] = ("Exploit", "User")
    v["Enterprise1"] = ("Scan", "User")
    assert most_suspicious_host(v, "Enterprise") == "Enterprise2"
    v["Enterprise1"] = ("Exploit", "User")             # now a full tie
    assert most_suspicious_host(v, "Enterprise") == "Enterprise1"


def test_resolve_action_global_and_targeted():
    """Correctness property: Sleep/Monitor have no target; subnet actions get
    the most suspicious host of THAT subnet only (Defender is never targeted)."""
    v = clean()
    v["Defender"] = ("Exploit", "Privileged")
    v["Op_Host2"] = ("Scan", "No")
    assert resolve_action(ACTION_LABELS.index("Sleep"), v) == ("Sleep", None)
    assert resolve_action(ACTION_LABELS.index("Monitor"), v) == ("Monitor", None)
    assert resolve_action(ACTION_LABELS.index("Restore-Operational"), v) == ("Restore", "Op_Host2")
    assert resolve_action(ACTION_LABELS.index("Remove-Enterprise"), v) == ("Remove", "Enterprise0")
