"""cyberq: a tabular Q-learning blue defender for CybORG CAGE Challenge 2.

Data flow (see docs/WALKTHROUGH.md):
    CybORG -> env_wrapper (BlueTable -> abstraction.abstract_state) -> state 0..17
           -> q_agent.select_action -> action 0..10
           -> actions.resolve_action (pick real host) -> CybORG
"""
