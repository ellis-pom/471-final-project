# Design decisions log

One entry per decision: **decision → reason → alternatives considered → how it could affect results.**
Entries marked *(sim finding)* record something we learned by reading or running CybORG.

---

### D1. Simulator: CybORG CAGE Challenge 2 (not MiniCAGE)
- **Decision:** Use the official CAGE 2 CybORG, vendored as a git submodule pinned to commit `26ce1c1`, on Python 3.9 in WSL2.
- **Reason:** It is the canonical environment with the official red agents and reward. It installed cleanly on Python 3.9 with `numpy==1.23.5` and `gym==0.26.2`. It is fast enough: about 0.04 s per 30-step episode, so a 500-episode training run takes about 20 s.
- **Alternatives:** MiniCAGE (CybORG++) is about 1000× faster but is a re-implementation. We'd only need it for very large sweeps.
- **Effect:** Results are directly comparable with CAGE 2 literature (same scenario, red agents and reward). CybORG's own `setup.py` doesn't pin versions, so we pin everything in `requirements.txt`. System Python 3.14 is too new for CybORG's dependencies.

### D2. Host groups *(sim finding)*
- **Decision:** User = `User0–4`, Enterprise = `Enterprise0–2`, Operational = `Op_Server0, Op_Host0–2`.
- **Reason:** Verified in `Scenario2.yaml` and in the printed blue table. These are the 13 hosts we expected, but **Defender sits in the Enterprise subnet**.
- **Effect:** See D3.

### D3. Exclude Defender from the Enterprise count and from targeting
- **Decision:** `enterprise_flagged` counts only Enterprise0–2, and Enterprise actions never target Defender.
- **Reason:** Defender is Blue's own server. Red's playbook never targets it, and Restoring it makes no tactical sense.
- **Alternatives:** Count all four hosts in the Enterprise subnet.
- **Effect:** None expected for B_line or Meander. It prevents a meaningless target.

### D4. Monitor runs automatically every step; we still keep the Monitor action *(sim finding)*
- **Decision:** Keep the 11-action space from the approved design, including `Monitor`.
- **Reason:** `EnvironmentController.step` executes a Blue `Monitor` after every step, whatever Blue chose. So the explicit Monitor action does the same as Sleep (both cost 0). We kept it to preserve the graded design.
- **Alternatives:** Drop Monitor and have 10 actions.
- **Effect:** It's a built-in sanity check: a well-trained agent should have Q(Sleep) ≈ Q(Monitor). It also spreads exploration over one redundant action.

### D5. Definition of "flagged"
- **Decision:** A host is flagged if `Activity != None` **or** `Compromised != No` (`abstraction.is_flagged`).
- **Reason:** It's the simplest rule that uses both columns of the blue table.
- **Facts behind it** *(sim finding, `BlueTableWrapper.py`)*:
  - `Activity` (None/Scan/Exploit) only shows what happened **this step**, and green users also generate Scans, which are false positives.
  - `Compromised` persists. It becomes `User` after an observed Exploit and `Privileged` only after **Analyse** finds red malware. Blue cannot see privilege escalation otherwise.
  - `Remove` sets `Compromised` to `Unknown`, which still counts as flagged. Only `Restore` sets it back to `No`.
  - Red's starting host User0 never shows as flagged, because red starts there without exploiting it.
- **Effect:** Remove never fully un-flags a host. Because `User` and `Privileged` count the same, Analyse can't change the abstract state; it only affects which host gets targeted next (D6). We expect the agent may learn that Analyse has little value.

### D6. "Most suspicious host" targeting rule
- **Decision:** Inside the chosen subnet, rank hosts by Compromised (Privileged > User > Unknown > No), then by Activity (Exploit > Scan > None), then by fixed host order. If nothing is flagged, target the first host in the list: User0, Enterprise0 or Op_Server0 (`actions.most_suspicious_host`).
- **Reason:** It's deterministic and explainable. Compromise ranks first because it is persistent evidence, while Activity may be a false positive.
- **Alternatives:** Random choice among flagged hosts, or letting the agent pick the host (which would multiply the action space).
- **Effect:** When nothing is flagged, subnet actions hit the default host. For User that's User0, red's permanent foothold, so `Analyse-User` there is mostly wasted.

### D7. Hyperparameters: α = 0.1, γ = 0.95, ε from 1.0 to 0.05 linearly over 80% of episodes, 500 episodes
- **Reason:**
  - **α = 0.1:** rewards are noisy (exploit success is random, green causes false positives), so we take small steps that average over samples.
  - **γ = 0.95:** gives an effective horizon of about 20 steps, roughly how long B_line takes to reach Op_Server0.
  - **ε schedule:** explore heavily at first, then exploit while still sampling a little.
  - **500 episodes:** about 20 s of training; enough for the checkpoint.
- **Alternatives:** Exponential ε decay, or α decaying over time. Both are planned for final tuning.
- **Effect:** See the first-run interpretation in PROGRESS.md. Q-value gaps between actions are tiny (often < 0.05), which is the same size as the update noise at α = 0.1.

### D8. Episodes are cut at a time limit, so we bootstrap at the final step
- **Decision:** CybORG never signals `done`, so the wrapper ends each episode after `max_steps` (30). `train.py` calls `update(..., terminal=False)` on **every** step, including the last one.
- **Reason:** Step 30 isn't a real end state, because the attack would continue. Our state contains no clock, so treating step 30 as terminal would give the same state two different targets: sometimes "the future is worth 0" and sometimes not. That would bias the Q-values. Bootstrapping at a time limit is the standard fix (Pardo et al., 2018, *Time Limits in Reinforcement Learning*).
- **Alternatives:** Treat step 30 as terminal (simpler, but inconsistent).
- **Effect:** Q-values estimate the value of continuing to defend indefinitely, discounted at γ. The agent class still supports `terminal=True`, and the toy-MDP test uses it.

### D9. Q-table initialised to zero; ties broken toward the lowest action index
- **Reason:** Zero initialisation is standard. Every CAGE 2 reward is ≤ 0, so an untried action (Q = 0) looks better than any tried one. This acts as *optimistic initialisation*: once the agent's exploration falls, its greedy choice still tries each action at least once in each visited state. Deterministic `argmax` tie-breaking keeps evaluation reproducible.
- **Alternatives:** Seeded random tie-breaking.
- **Effect:** A state never visited in training keeps an all-zero row, so it defaults to **Sleep** (action 0). `inspect_policy` marks these states. This matters for generalisation: see the Meander teaser in PROGRESS.md.

### D10. Evaluation protocol
- **Decision:** Evaluation uses seed 100 (training used 0, 1 and 2), runs the greedy policy (ε = 0) with no updates, and gives every agent a fresh environment with the same seed. Baselines are Sleep, Random and RestoreOnly.
- **Reason:** These are identical conditions, separate from training. The code asserts that the Q-table is unchanged after evaluation.
- **Note:** B_line is deterministic. Variation between episodes comes from green activity, random exploit success, and randomness in Blue actions (Analyse and Remove pick a session at random).
- **RestoreOnly baseline:** it restores the most critical flagged subnet (Operational, then Enterprise, then User) and otherwise Sleeps. It's a plausible "human rule", not a straw man.

### D11. Evaluation metrics read the simulator's ground truth
- **Decision:** "Red hosts at end", "red privileged at end" and "OpSrv lost" come from CybORG's **true** state (`AbstractDefenderEnv.red_footprint`). The agent never sees this.
- **Reason:** These are operational metrics: did we lose the server? They don't depend on Blue's partial view.
- **Effect:** User0 is always counted, because it's red's foothold.
