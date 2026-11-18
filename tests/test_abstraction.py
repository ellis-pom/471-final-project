"""State abstraction: flagged definition, binning, encoding."""
import pytest

from cyberq.abstraction import (ENTERPRISE_HOSTS, N_STATES, OPERATIONAL_HOSTS, USER_HOSTS,
                                abstract_state, count_to_bin, decode_state, encode_state,
                                is_flagged, state_label)

ALL_HOSTS = USER_HOSTS + ENTERPRISE_HOSTS + OPERATIONAL_HOSTS + ["Defender"]


def clean():
    """A blue view where nothing is flagged."""
    return {h: ("None", "No") for h in ALL_HOSTS}


def test_flagged_definition():
    """Correctness property: a host is flagged iff Activity != None OR Compromised != No
    (so a Remove'd host, Compromised=Unknown, is still flagged)."""
    assert not is_flagged("None", "No")
    assert is_flagged("Scan", "No")
    assert is_flagged("Exploit", "User")
    assert is_flagged("None", "Unknown")
    assert is_flagged("None", "Privileged")


def test_count_bins_cap_at_two():
    """Correctness property: counts map to bins 0, 1, 2+ and every count >= 2 lands in the 2+ bin."""
    assert [count_to_bin(n) for n in range(6)] == [0, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError):
        count_to_bin(-1)


def test_encode_decode_round_trip_all_18_states():
    """Correctness property: encode/decode is a bijection between the 18 bin
    combinations and the indices 0..17 (no collisions, nothing missing)."""
    seen = set()
    for u in range(3):
        for e in range(3):
            for o in range(2):
                s = encode_state(u, e, o)
                assert decode_state(s) == (u, e, o)
                seen.add(s)
    assert seen == set(range(N_STATES))
    assert state_label(17) == "User:2+ | Ent:2+ | OpSrv:yes"


def test_synthetic_observations_map_to_expected_state():
    """Correctness property: hand-built blue views produce the expected bins and index."""
    v = clean()
    assert abstract_state(v) == 0                                  # all quiet

    v["User2"] = ("Scan", "No")                                    # 1 user flagged
    assert decode_state(abstract_state(v)) == (1, 0, 0)

    for h in ("User1", "User3", "User4"):                          # 4 users flagged -> 2+
        v[h] = ("None", "User")
    v["Enterprise1"] = ("Exploit", "User")                         # 1 enterprise
    v["Op_Server0"] = ("None", "Privileged")                       # op server
    assert decode_state(abstract_state(v)) == (2, 1, 1)
    assert abstract_state(v) == 2 * 6 + 1 * 2 + 1 == 15


def test_defender_and_op_hosts_do_not_change_state():
    """Correctness property: Defender is excluded from the Enterprise count and
    Op_Host0-2 do not affect the op-server bit (only Op_Server0 does)."""
    v = clean()
    v["Defender"] = ("Exploit", "Privileged")
    v["Op_Host1"] = ("Exploit", "Privileged")
    assert abstract_state(v) == 0
