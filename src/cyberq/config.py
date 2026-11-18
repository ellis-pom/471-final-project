"""Default hyperparameters in one place. Every value can be overridden by a CLI flag.

Why these values (see DECISIONS.md D7):
  ALPHA 0.1   - small steps; CAGE 2 rewards are noisy (green false positives,
                random exploit success), so we average over many samples.
  GAMMA 0.95  - effective horizon ~1/(1-0.95) = 20 steps, about the time B_line
                needs to reach Op_Server0, so the agent can "see" that threat coming.
  EPS 1.0 -> 0.05 linearly over 80% of training - explore everything early,
                then mostly exploit while still sampling a little.
"""
ALPHA = 0.1
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.8   # epsilon reaches EPS_END after this fraction of episodes

TRAIN_EPISODES = 500
MAX_STEPS = 30             # checkpoint episode length (design also mentions 50/100)
RED = "bline"
TRAIN_SEED = 0
EVAL_SEED = 100            # evaluation uses DIFFERENT seeds from training
EVAL_EPISODES = 50
