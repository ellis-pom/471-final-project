# Checkpoint Progress Summary

**Team:** Ellis Pomales III, William Kim, Brendan Gibbons

## Completed functionality
- CybORG CAGE 2 installed and pinned (Python 3.9, WSL2), plus a smoke script that prints the raw blue observation.
- A wrapper that turns the blue table into **18 abstract states** and the **11 abstract actions** into real CybORG actions, using a documented "most suspicious host" rule.
- A tabular Q-learning agent with ε-greedy selection, the TD update, linear ε decay, save/load (Q-table + visit counts + JSON metadata), and a greedy evaluation mode (ε = 0, no updates).
- Baselines (Sleep, Random, RestoreOnly), training and evaluation CLIs, a policy inspector, per-step episode traces and action-distribution reports.
- **22 pytest tests**. Among them: hand-computed TD updates, abstraction and targeting rules, save/load, a toy MDP where the agent recovers the known optimal policy and Q*, and integration tests against the real simulator.

## Initial results (500 training episodes vs B_line; eval: 50 greedy episodes, held-out seed 100)

| Agent | Mean return ± std | Op_Server lost |
|---|---|---|
| Q-agent, train seeds 0 / 1 / 2 | −61.5 ± 37.6 / −44.3 ± 53.6 / −17.3 ± 7.4 | 8% / 18% / 0% |
| Random | −59.3 ± 37.6 | 64% |
| Sleep | −217.1 ± 17.8 | 100% |
| RestoreOnly (scripted) | −15.9 ± 0.5 | 0% |

The agent trains, saves and runs a greedy policy end to end. It clearly beats Sleep, but it is **not yet reliably better than Random**, and it is worse than the simple RestoreOnly rule. Results swing a lot by training seed. The Q-values of different actions in the same state differ by only about 0.01–0.5, so which action "wins" is fragile. Only 12 of 18 states were ever visited: B_line never flags 2+ user hosts. We make **no convergence claim**.

## Remaining work
More episodes and seeds, plus tuning of α and ε. Convergence plots across seeds. The generalisation study against Meander. An early teaser: the B_line-trained agent scores −39.2, the same as Sleep (−39.9), because Meander drives it into states it never visited. Operational metrics, and the final analysis.

## Challenges
- System Python 3.14 was too new, and CybORG doesn't pin its dependencies.
- In the simulator, Monitor runs automatically every step, and privilege escalation is only visible after Analyse.
- Episodes never end on their own, which forced a decision about bootstrapping at the time limit.

## Member contributions
The work was split evenly among the three of us, with each member owning a different part:
- **Ellis Pomales III:** set up the repository and environment, and built the implementation (wrapper, Q-learning agent, baselines, CLIs, tests, docs) with the help of an AI coding assistant (Claude Code), which the course allows.
- **William Kim:** researched the project area and identified instructors we could contact for additional help.
- **Brendan Gibbons:** cross-checked alternative project ideas, and reviewed our work against the checkpoint rubric.
