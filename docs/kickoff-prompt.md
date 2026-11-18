# CS471 Final Project — Checkpoint Kickoff Prompt for Claude Code

> Paste everything below this line into Claude Code from the root of our GitHub repo.

---

## Who I am and what we're building

I'm Ellis, a Computer Science / Cyber Science cadet. My 3-person team (me + a data science major + an operations research major) is building our CS471 (Artificial Intelligence) final project: a **tabular Q-learning cyber-defense agent** that defends a simulated network inside the **CybORG CAGE Challenge 2** simulator (or its fast lightweight re-implementation, **MiniCAGE**, from the CybORG++ project). Our instructor approved using the existing simulator as long as the agent, wrapper, abstraction, and experiments are our own work.

We are on Windows 11. I need to **understand and be able to defend every line we submit**, so explain design decisions as you go, keep code readable, and add docstrings that explain the *why*, not just the *what*.

The long-term research question (final project, not this checkpoint) is whether a learned defender policy **generalizes to an attacker it never trained on**. Keep the code structured so the state abstraction and the red agent are easy to swap later — but do **not** build that now.

---

## What the final grade actually rewards (shape everything around this)

Our instructor grades the final on: **the problem, the technique, the structure, whether we understand the code and reasoning and can discuss it, overall application and understanding, and interpreting our results.** A high score alone earns little. We need to be able to say *why* the agent behaves the way it does, *what* it learned, *where* it fails, and *what that means tactically*.

So for every piece you build:
- Prefer **transparent over clever.** Simple loops we can trace by hand beat compact abstractions.
- Build **inspection tools** (see "Interpretation tooling" below) alongside the training code, not as an afterthought.
- Record **every design decision and its reason** in `DECISIONS.md` as we make it (why these bins, why this "flagged" definition, why this host-selection rule, why these α/γ/ε values, what we considered and rejected). This becomes the "structure and reasoning" section of our presentation.
- Never present a number without context: compare against baselines, show the spread across episodes/seeds, and say what the number means operationally.

---

## The approved design (graded 24.5/25 — do not change it without asking me)

**Operational problem:** A blue defender agent protects an enterprise network from an automated red attacker trying to reach the critical operational server, while keeping services available.

**States (18 total, discrete):** an abstraction of the blue observation:
- `user_flagged` — number of flagged hosts in the User subnet: 0 / 1 / 2+  (3 bins)
- `enterprise_flagged` — number of flagged hosts in the Enterprise subnet: 0 / 1 / 2+  (3 bins)
- `opserver_flagged` — does the Operational Server show malicious activity / compromise: no / yes  (2 bins)
- 3 × 3 × 2 = **18 states**, encoded as a single integer index 0–17.

**Actions:** subnet-level defensive actions — Monitor, Analyse, Remove, Restore, Sleep. Our wrapper applies Analyse / Remove / Restore to the **most suspicious host in the chosen subnet**. Proposed concrete action space (confirm with me):
- `Sleep`, `Monitor` (global, no target)
- `{Analyse, Remove, Restore} × {User, Enterprise, Operational}`
- → **11 discrete actions**

**Observations:** partial and noisy — a green (benign user) agent creates false-positive activity.

**Reward:** the simulator's default CAGE 2 reward (penalties for compromised hosts, heavier for servers / op server, cost for Restore). Pass it through unchanged.

**Episodes:** fixed length — 30 steps for the checkpoint (design also mentions 50 / 100 for later).

**Algorithm:** tabular Q-learning (off-policy TD control), ε-greedy with decaying ε:
`Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]`

---

## Instructor feedback on the design (follow this)

> For the checkpoint, demonstrate that the agent **completes training episodes, updates and saves its Q-table, and executes its learned policy during separate evaluation episodes without exploration**. Reporting **initial results against a scripted baseline is encouraged**. You do **not** need to demonstrate superior performance or reward convergence at the checkpoint.

---

## Checkpoint requirements (50 pts) — "Ugly but Functional"

| Criterion | Pts | What it means for us |
|---|---|---|
| Core algorithm implementation | 20 | Real Q-learning loop: ε-greedy selection, TD update, ε decay, Q-table save/load |
| Functional demonstration | 10 | Train on ≥1 representative scenario (CAGE 2 + one red agent) with observable terminal output |
| Basic correctness verification | 10 | **≥2 meaningful tests / correctness checks** (we'll do more) |
| GitHub repo & reproducibility | 5 | README with setup, dependencies, exact run commands; tagged commit |
| Team progress & accountability | 5 | Progress summary (~150–250 words): done, remaining, each member's contributions, challenges |

No GUI. Command line is fine. AI assistants and external libraries are allowed.

**Submission:** repo link with a **checkpoint git tag**, plus the progress summary (also committed to the repo). Repo must be public (I'll handle visibility).

---

## Step 0 — Before writing code, check in with me

1. Look at the current repo contents and tell me what's there.
2. Ask me: teammate names, Python version I have, and whether I want to use **WSL (Ubuntu)** or native Windows. (Recommend WSL if CybORG has Linux-only dependencies.)
3. Then propose the plan below adjusted to what you found, and wait for my OK.

---

## Step 1 — Get the simulator running (verify, don't assume)

- Find the official CAGE Challenge 2 repo (`cage-challenge/cage-challenge-2` on GitHub) and the CybORG++ / MiniCAGE repo. Check their READMEs for install steps and Python version requirements.
- **Decision rule:** Prefer **CAGE 2 CybORG** if it installs cleanly (it's the canonical environment and has official baselines). Use **MiniCAGE** if CybORG is broken or too slow (MiniCAGE is reported to be ~1000× faster). If you switch, tell me why. Vendor or pin the dependency so a fresh clone reproduces our setup.
- Write a tiny smoke script that resets the env, steps it a few times with a Sleep action, and **prints the raw blue observation** so we can see the real format.
- **Verify against the real code — these are my expectations, not facts:**
  - Host names (I expect something like `User0–4`, `Enterprise0–2`, `Op_Host0–2`, `Op_Server0`, `Defender`).
  - The blue observation format (I expect a per-host table with an *Activity* field like None/Scan/Exploit and a *Compromised* field like No/Unknown/User/Privileged — e.g., via a `BlueTableWrapper`).
  - Red agent class names (I expect `B_lineAgent` and `RedMeanderAgent`).
  - Blue action class names and their parameters (`Sleep`, `Monitor`, `Analyse`, `Remove`, `Restore`, and whether Monitor is implicit each step).
  - Whether the reward is returned per step from blue's perspective.
- Report back what you found **before** building the wrapper. If reality differs from the design (e.g., observations don't support "flagged" the way I described), propose the smallest change that preserves the 18-state design and explain it to me.

---

## Step 2 — Build it (suggested structure; adapt if needed)

```
src/cyberq/            (or similar package name)
  env_wrapper.py       # AbstractDefenderEnv: raw obs -> state 0..17, action 0..10 -> real CybORG action
  abstraction.py       # pure functions: flagged-host counting, binning, state encode/decode
  actions.py           # action table + "most suspicious host" selection rule
  q_agent.py           # QLearningAgent: select_action (ε-greedy / greedy), update, decay, save/load
  baselines.py         # RandomAgent, SleepAgent, RestoreOnlyAgent (same interface as QLearningAgent)
  train.py             # CLI: train agent, log per-episode stats, save Q-table + metadata
  evaluate.py          # CLI: run greedy policy (ε=0) + baselines on separate eval episodes, print table
  config.py            # defaults: α, γ, ε schedule, episodes, steps, seed, red agent
tests/
  ...                  # pytest
results/               # gitignored except a small sample run we commit for the checkpoint
README.md
PROGRESS.md            # checkpoint progress summary
requirements.txt
```

### Wrapper details
- `reset() -> state:int`, `step(action:int) -> (state:int, reward:float, done:bool, info:dict)`. The agent never touches CybORG directly.
- Keep the observation→state logic as **pure functions** in `abstraction.py` so they're unit-testable without the simulator.
- Define **"flagged"** explicitly in one place (proposal: a host is flagged if its Activity ≠ None **or** its Compromised ≠ No) and document it in the README.
- **Most suspicious host rule** (proposal — deterministic, documented): rank by Compromised (Privileged > User > Unknown > No), then Activity (Exploit > Scan > None), then a fixed host-order tiebreak. If nothing in the subnet is flagged, target a fixed default host for that subnet (e.g., `Op_Server0` for the Operational subnet). Log the chosen real host in `info` so we can inspect behavior.
- Put the chosen real action in `info` too.

### Q-learning agent
- Q-table: NumPy array shape `(18, 11)`, initialized to 0.
- ε-greedy with deterministic tie-breaking (or seeded random tie-breaking — pick one and document it).
- Linear or exponential ε decay from 1.0 to a floor (e.g., 0.05). Sensible starting defaults: α = 0.1, γ = 0.95; expose all hyperparameters as CLI flags.
- `greedy=True` mode for evaluation: **ε = 0, no Q-table updates**.
- `save(path)` / `load(path)`: Q-table as `.npy` plus a JSON sidecar with hyperparameters, seed, red agent, episode count, state/action label lists, and a timestamp.
- Track visit counts per (state, action) — useful for showing which states the agent actually sees.

### Training CLI (example — adjust names)
```
python -m cyberq.train --episodes 500 --steps 30 --red bline --seed 0 --out results/run_bline_s0
```
Per-episode terminal line: episode #, total reward, ε, # distinct states visited so far, # nonzero Q entries.
Also write `train_log.csv`. Optional: `--plot` saves a reward-per-episode PNG with a moving average (matplotlib). Keep it optional.

### Evaluation CLI
```
python -m cyberq.evaluate --qtable results/run_bline_s0/qtable.npy --episodes 50 --steps 30 --red bline --seed 100
```
- Uses **different seeds** than training.
- Runs the greedy Q-agent **and** RandomAgent, SleepAgent, RestoreOnlyAgent under identical conditions.
- Prints a table: mean reward ± std, and if easy, mean # hosts compromised at episode end and # Restore actions.
- Writes `eval_results.csv`.
- Optional teaser (only if trivial): also evaluate against the other red agent. Do **not** build the full generalization study yet.

### Interpretation tooling (small, but build it now)
These let us *explain* the agent instead of just scoring it. Keep each one simple.
- **Policy table** — `python -m cyberq.inspect_policy --qtable ...` prints all 18 states with human-readable labels (e.g., `User:2+ | Ent:1 | OpSrv:yes`), the greedy action, its Q-value, the runner-up action and the gap between them, and the visit count. Mark never-visited states clearly (their "policy" is just the zero-initialized default and must not be interpreted). Also export as CSV.
- **Q-table export** — CSV with state labels as rows and action labels as columns, so teammates can open it in Excel or pandas.
- **Episode trace** — `--trace` flag on evaluate that prints one episode step by step: abstract state → chosen action → real host targeted → reward → what red did (if the simulator exposes it). This is how we narrate "what the defender did" in the presentation.
- **Action distribution** — in evaluation, report how often each action was used per agent (Q-agent vs. baselines). This answers questions like "did it just learn to spam Restore?"
- **State coverage** — which of the 18 states were visited during training, and how often. If the agent only ever sees 5 states, that's a finding we need to be able to explain.

These feed the final-project questions: Does the learned policy make tactical sense? Where does it disagree with a human's intuition? Which states does it never learn about? What does it do differently from the baselines?

### Reproducibility
- Seed Python `random`, NumPy, and the simulator wherever it accepts a seed.
- Pin versions in `requirements.txt`.

---

## Step 3 — Tests (rubric needs ≥2; aim for these)

Use `pytest`. Each test needs a short docstring stating **what correctness property it checks** — I'll quote these in the report.

1. **Q-update math:** hand-computed single update (given Q, s, a, r, s', α, γ) matches the agent's result exactly; a terminal transition ignores the bootstrap term.
2. **State abstraction:** synthetic observations → expected bins and index; encode/decode round-trips for all 18 states; "2+" bin caps correctly.
3. **Action mapping:** most-suspicious-host rule picks the expected host on hand-built cases, including the "nothing flagged" default and ties.
4. **Greedy policy:** with ε = 0 the agent always returns argmax and does **not** modify the Q-table during evaluation.
5. **Save/load round-trip:** saved then loaded Q-table and metadata are identical.
6. **Learning sanity check on a toy MDP (strongest correctness evidence):** a tiny hand-made environment with a known optimal policy (e.g., a 3–4 state chain), using the same agent class — after training, the greedy policy equals the known optimum. Keeps the test fast and independent of CybORG.
7. **Integration smoke test (mark as slow if needed):** wrapper + real simulator run 2 short episodes without crashing; states stay in 0–17, actions in 0–10.

---

## Step 4 — Docs

**README.md** must include: project summary (2–3 sentences), the state/action/reward design table, what "flagged" and "most suspicious host" mean, setup (Windows/WSL steps, Python version, `pip install -r requirements.txt`, simulator install), exact commands to train / evaluate / run tests, expected sample output, and repo layout.

**DECISIONS.md** — running log, one short entry per decision: *decision → reason → alternatives considered → how it could affect results.* Include anything surprising we learned about the simulator.

**PROGRESS.md** (~150–250 words, can be longer) — draft it but **leave teammate contributions as clearly marked placeholders and ask me for them; don't invent who did what.** Sections:
- Completed functionality
- Initial results (training run + eval vs. baselines — honest numbers, no claims of convergence)
- Remaining work (more episodes/seeds, convergence plots, generalization test vs. unseen attacker, operational metrics, final analysis)
- Challenges encountered (whatever actually happened: install issues, obs format surprises, etc.)
- Member contributions (placeholders)

---

## Step 5 — Git

- Small, meaningful commits as we go.
- **Ask me before pushing, and before creating the checkpoint tag.** When I approve: `git tag -a checkpoint -m "CS471 Lesson 36 checkpoint"` and push the tag.

---

## Working rules

- **Stay in scope.** No deep RL, no GUI, no extra state abstractions, no reward shaping. Those come later.
- If something in the design turns out infeasible, **stop and ask me** with a proposed minimal fix.
- After each step, give me a short plain-English summary of what the code does and anything I'd need to explain if my instructor asks.
- When done, give me a walkthrough of the Q-learning loop referencing exact file/function names, so I can present it.
- After the first real training + eval run, help me **interpret** it, not just report it: read the policy table and an episode trace with me, point out what looks tactically sensible vs. suspicious (e.g., spamming one action, never-visited states, ties at Q=0), and note what questions the final experiments should answer.
- Write a short `docs/WALKTHROUGH.md` my teammates can read to understand the code end to end — one data-flow diagram in text (simulator → wrapper → state → agent → action → wrapper → simulator) plus a plain-English explanation of each file.
