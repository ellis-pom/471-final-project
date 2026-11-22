# CS471 Final Project: Tabular Q-Learning Cyber Defender (CybORG CAGE 2)

A **tabular Q-learning** blue agent defends a simulated enterprise network inside the
[CybORG CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2) simulator. An
automated red attacker tries to reach and disrupt the Operational Server. Our contribution is the
wrapper, the 18-state abstraction, the agent, the baselines and the experiments; the simulator is
used unchanged. Longer-term question: does a learned defender **generalise to an attacker it
never trained on**?

## Design

| Element | Design |
|---|---|
| **States (18)** | `user_flagged` ∈ {0, 1, 2+} × `enterprise_flagged` ∈ {0, 1, 2+} × `opserver_flagged` ∈ {no, yes}, encoded as `index = user*6 + ent*2 + op` |
| **Actions (11)** | `Sleep`, `Monitor`, and `{Analyse, Remove, Restore} × {User, Enterprise, Operational}`. Subnet actions hit the *most suspicious host* in that subnet |
| **Reward** | Default CAGE 2 Blue reward, unchanged: −0.1 per user host and −1 per enterprise host / Op_Server0 where red has root, −10 per step red Impacts Op_Server0, −1 per Restore |
| **Episodes** | Fixed length, 30 steps (`--steps`) |
| **Algorithm** | Q-learning: `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`, ε-greedy with linear ε decay. Defaults: α = 0.1, γ = 0.95, ε 1.0 → 0.05 over 80% of episodes |
| **Red agents** | `bline` (B_lineAgent: goes straight for Op_Server0), `meander` (RedMeanderAgent: spreads through every host) |

**Flagged host:** a host is flagged if its blue-table `Activity` ≠ `None` **or** `Compromised` ≠ `No`
(`src/cyberq/abstraction.py::is_flagged`). Activity only lasts one step and can be a green false
positive; Compromised persists until Restore.

**Most suspicious host:** inside the chosen subnet, rank by Compromised (Privileged > User > Unknown > No),
then Activity (Exploit > Scan > None), then fixed host order. If nothing is flagged, target
User0 / Enterprise0 / Op_Server0 (`src/cyberq/actions.py::most_suspicious_host`). The
`Defender` host is never counted or targeted.

The reasoning behind every choice is in [DECISIONS.md](DECISIONS.md). For a code walkthrough, see
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Setup

Tested on **Windows 11 + WSL2 (Ubuntu)** with **Python 3.9**. CybORG needs an old Python (≤ 3.10); a
newer system Python will not work. On Windows, install WSL first (`wsl --install` in an admin
PowerShell), then do everything below inside the Ubuntu terminal.

```bash
# 1. clone WITH the pinned CybORG submodule
git clone --recurse-submodules https://github.com/ellis-pom/471-final-project.git
cd 471-final-project
#    (already cloned without it?  git submodule update --init)

# 2. install uv (manages the Python 3.9 interpreter without sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc            # or open a new terminal so `uv` is on PATH

# 3. create the venv and install everything (CybORG + our package, editable)
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` pins every version (`numpy==1.23.5`, `gym==0.26.2`, ...) and installs CybORG from
`external/cage-challenge-2/CybORG` (commit `26ce1c1`). Gym prints a "Gym has been unmaintained"
notice on import; it's harmless. Set `export PYTHONWARNINGS=ignore` to hide it.

## Run

```bash
# see the raw simulator output (blue table, red actions, rewards)
python scripts/smoke_env.py --steps 8

# train (about 20 s); writes qtable.npy, visits.npy, meta.json, train_log.csv, reward_curve.png
python -m cyberq.train --episodes 500 --steps 30 --red bline --seed 0 --out results/sample_bline_s0 --plot

# evaluate the greedy policy (epsilon = 0, no learning) vs Random / Sleep / RestoreOnly on held-out seed 100
python -m cyberq.evaluate --qtable results/sample_bline_s0/qtable.npy --episodes 50 --steps 30 --red bline --seed 100 --trace

# explain the policy: per-state greedy action, runner-up, gap, visit counts; writes policy.csv, qtable.csv, visits.csv
python -m cyberq.inspect_policy --qtable results/sample_bline_s0/qtable.npy

# teaser: the B_line-trained agent against the attacker it never saw
python -m cyberq.evaluate --qtable results/sample_bline_s0/qtable.npy --red meander --seed 100 --out results/sample_bline_s0/eval_vs_meander

# tests (22 tests, about 1 s; `-m "not slow"` skips the 3 that run the real simulator)
python -m pytest -v
```

Runs are deterministic for a given seed (CybORG uses Python's global `random`, which we seed).

### Expected sample output (the committed run, `results/sample_bline_s0/`)

```
episode   return epsilon states_seen nonzero_Q mean|TD|
      1    -39.1   1.000        8/18     23/198    1.277
    250    -18.3   0.409       12/18    132/198    0.618
    500    -41.2   0.050       12/18    132/198    0.992

agent         mean return    std     min     max restores/ep red hosts@end red priv@end OpSrv lost
Q-agent             -61.5   37.6  -206.8   -15.7         9.1           4.1          3.9        8%
Random              -59.3   37.6  -202.1   -22.8         8.2           2.6          2.6       64%
Sleep              -217.1   17.8  -224.8  -163.7         0.0           5.0          5.0      100%
RestoreOnly         -15.9    0.5   -16.4   -13.7        14.8           1.1          1.1        0%
```

See [PROGRESS.md](PROGRESS.md) for what these numbers mean.

## Correctness checks (what each test proves)

`python -m pytest -v` runs 22 tests (**22 passed** at the `checkpoint` tag, about 1 s). Each test's docstring
states the property it checks. Summary:

| # | Test (file) | What it checks | Expected behavior | Result |
|---|---|---|---|---|
| 1 | Q-update math (`test_q_update.py`) | One `update()` call applies exactly `Q ← Q + α[r + γ·max Q(s′,·) − Q]` | Hand calculation: Q(0,1)=2.0, max Q(1,·)=5, r=−1, α=0.1, γ=0.9 → target 3.5, TD error 1.5, new Q = **2.15** | Pass: agent gives 2.15 |
| 2 | Terminal update (`test_q_update.py`) | At the end of an episode there is no future value | Target is just r; Q(s′,·) is ignored | Pass |
| 3 | Greedy mode (`test_q_update.py`) | Evaluation never explores | With `greedy=True`, even at ε = 1.0 the agent always returns argmax Q (ties → lowest index) | Pass |
| 4 | Exploration (`test_q_update.py`) | Training mode really explores | At ε = 1 the choice is uniform: a non-greedy action about half the time | Pass |
| 5 | ε decay (`test_q_update.py`) | The exploration schedule | ε falls linearly from 1.0 to the 0.05 floor, then stays there | Pass |
| 6 | **Toy MDP** (`test_toy_mdp.py`, seeds 0/1/2) | The whole loop (select → update → decay) actually *learns*, independent of CybORG | 4-state corridor, −1 per step, γ = 0.9. Optimal policy is "always right", with Q\*(0,R) = −2.71, Q\*(1,R) = −1.9, Q\*(2,R) = −1.0 | Pass: greedy policy = optimum and Q within 0.05 of Q\* for all 3 seeds |
| 7 | Save/load (`test_save_load.py`) | A saved agent comes back identical | Q-table, visit counts, hyperparameters, ε, episode count and JSON metadata all round-trip exactly | Pass |
| 8 | "Flagged" rule (`test_abstraction.py`) | The definition the whole state space rests on | Flagged ⇔ Activity ≠ None **or** Compromised ≠ No (a host after Remove, Compromised = Unknown, stays flagged) | Pass |
| 9 | Binning (`test_abstraction.py`) | Counts become 0 / 1 / 2+ | Every count ≥ 2 lands in the 2+ bin | Pass |
| 10 | Encode/decode (`test_abstraction.py`) | The 18 states are a clean numbering | All 18 bin combinations ↔ indices 0–17, no collisions, nothing missing | Pass |
| 11 | Synthetic observations (`test_abstraction.py`) | Real-shaped blue tables map to the right state | Hand-built tables give the expected bins and index; Defender and Op_Host0–2 don't change the state | Pass (2 tests) |
| 12 | Action table (`test_actions.py`) | The 11 approved actions, in order | Sleep, Monitor, then {Analyse, Remove, Restore} × {User, Enterprise, Operational} | Pass |
| 13 | Most-suspicious-host rule (`test_actions.py`) | The wrapper hits the right machine | Compromise level beats Activity; then Exploit > Scan; then host order; nothing flagged → User0 / Enterprise0 / Op_Server0; Defender is never targeted | Pass (4 tests) |
| 14 | Real simulator (`test_integration.py`, `slow`) | Wrapper + CybORG work end to end | 2 episodes run without crashing; states stay 0–17, actions 0–10 accepted, episode ends at exactly `max_steps`, rewards ≤ 0 | Pass |
| 15 | Reproducibility (`test_integration.py`, `slow`) | Same seed gives the same run | Same seed + same actions → identical rewards and states | Pass |
| 16 | Evaluation doesn't learn (`test_integration.py`, `slow`) | Separate evaluation episodes use the frozen policy | Greedy episodes against the real simulator leave the Q-table and visit counts unchanged | Pass |

Tests 1, 6 and 16 are the core evidence for the checkpoint: the update rule is mathematically right, the agent
converges to a known optimum when one exists, and evaluation runs the learned policy without exploring or learning.

## Repository layout

```
src/cyberq/
  abstraction.py    pure functions: flagged rule, binning, state encode/decode/labels
  actions.py        11-action table + most-suspicious-host rule (pure)
  env_wrapper.py    AbstractDefenderEnv: the only code that touches CybORG
  q_agent.py        QLearningAgent: epsilon-greedy, TD update, epsilon decay, save/load, visit counts
  baselines.py      SleepAgent, RandomAgent, RestoreOnlyAgent
  config.py         default hyperparameters
  train.py          training CLI
  evaluate.py       evaluation CLI (baselines, action distribution, --trace)
  inspect_policy.py policy table, Q-table/visits CSV export, state coverage
tests/              pytest: Q-update math, abstraction, action rule, save/load, toy MDP, integration
scripts/smoke_env.py  prints the raw simulator observation
external/cage-challenge-2/  CybORG CAGE 2 (git submodule, pinned)
results/sample_bline_s0/    committed checkpoint run (other results/ are gitignored)
DECISIONS.md  PROGRESS.md  docs/WALKTHROUGH.md
```
