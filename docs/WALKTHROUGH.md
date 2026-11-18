# Code walkthrough (for the team)

## Data flow in one step

```
            ┌──────────────────────── CybORG (CAGE 2 simulator) ────────────────────────┐
            │  Red agent (B_line / Meander) + Green users + Blue's chosen action act      │
            │  Reward for Blue computed; Monitor runs automatically for Blue              │
            └───────────────┬────────────────────────────────────────▲──────────────────┘
                            │ BlueTableWrapper: 13-row table          │ real action, e.g.
                            │ (Hostname, Activity, Compromised)       │ Restore(hostname="Op_Server0")
                            ▼                                         │
   env_wrapper.AbstractDefenderEnv.step()                             │
     abstraction.table_to_host_status()   {host: (activity, compromised)}
     abstraction.abstract_state()         count flagged per subnet -> bins -> state 0..17
                            │                                         │
                            ▼                                         │
               (state, reward, done, info)          env_wrapper.make_cyborg_action(verb, host)
                            │                                         ▲
                            ▼                                         │
   q_agent.QLearningAgent                                actions.resolve_action(a, host_status)
     update(s, a, r, s', terminal=False)  <- learn        "Restore-Operational" -> most suspicious
     select_action(s')  epsilon-greedy    -> action 0..10    host in Operational subnet
                            └─────────────────────────────────────────┘
```

The agent only ever sees integers: a state index 0–17, an action index 0–10, and a float reward.
Everything about networks and hosts stays in the wrapper, the abstraction and the actions modules.

## The Q-learning loop (`src/cyberq/train.py::train`)

```python
state = env.reset()
while not done:
    action = agent.select_action(state)                 # epsilon-greedy
    next_state, reward, done, _ = env.step(action)      # act in CybORG
    agent.update(state, action, reward, next_state, terminal=False)
    state = next_state
agent.end_episode()                                     # decay epsilon
```

1. **`QLearningAgent.select_action`** (`q_agent.py`): with probability ε it picks a uniformly random action (explore); otherwise it picks `argmax Q[state]` (exploit), with ties going to the lowest index. With `greedy=True` (evaluation), ε is ignored.
2. **`AbstractDefenderEnv.step`** (`env_wrapper.py`): turns the action into a real host, runs one CybORG step, re-reads the blue table and computes the new state.
3. **`QLearningAgent.update`**: `target = r + γ·max Q[s']`, `Q[s,a] += α·(target − Q[s,a])`, and adds one to `visits[s,a]`. We pass `terminal=False` even on the last step, because the 30-step cutoff is a time limit, not a real end (DECISIONS D8).
4. **`QLearningAgent.end_episode`**: ε moves linearly toward 0.05.
5. After training, **`save`** writes `qtable.npy`, `visits.npy` and `meta.json`.

## Each file in plain English

| File | What it does |
|---|---|
| `abstraction.py` | Defines "flagged" and turns the 13-host blue table into one number from 0 to 17. Pure functions, so everything is tested by hand. Also provides the state labels like `User:1 \| Ent:2+ \| OpSrv:yes`. |
| `actions.py` | The list of 11 actions, plus the rule for which host "Restore-Enterprise" actually hits (the most suspicious one, or a fixed default). |
| `env_wrapper.py` | The bridge to CybORG. It builds the simulator with the chosen red agent, seeds it, translates actions, ends the episode at 30 steps, and reports ground-truth red presence for evaluation metrics only. |
| `q_agent.py` | The learner: Q-table (18×11), visit counts, ε-greedy, TD update, ε decay, save/load. It knows nothing about cyber. |
| `baselines.py` | Comparison policies that use the same 18 states: Sleep (do nothing), Random, and RestoreOnly (restore the most critical flagged subnet). |
| `config.py` | Default hyperparameters and seeds in one place. |
| `train.py` | Runs the loop above, prints progress, writes `train_log.csv`, and optionally plots the reward curve. |
| `evaluate.py` | Runs the trained policy greedily (no learning) and the baselines on the same held-out seed. Prints mean ± std return, restores, red footprint and the action distribution. `--trace` narrates one episode step by step. |
| `inspect_policy.py` | Explains the Q-table: for each state it shows the best action, the runner-up, the gap between them and the number of visits, and marks never-visited states. Exports CSVs for Excel or pandas. |
| `scripts/smoke_env.py` | Prints the raw simulator view, so we can check what Blue actually sees. |
| `tests/` | `test_q_update` (TD math, greedy, ε decay), `test_abstraction`, `test_actions`, `test_save_load`, `test_toy_mdp` (the agent solves a 4-state corridor with a known optimum), `test_integration` (real simulator, reproducibility, evaluation doesn't learn). |

## How to read the outputs

- **`train_log.csv`**: one row per episode with return, ε, number of states seen, number of nonzero Q entries, and mean |TD error|.
- **`policy.csv`**: if `state_visits` = 0, that row means nothing. A small `gap` means the agent hardly prefers its choice.
- **`eval_results.csv`**: one row per agent per episode. Use it for box plots and spread.
- **`action_distribution.csv`**: answers "did it just spam one action?"
