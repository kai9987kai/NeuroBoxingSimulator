# NeuroBoxingSimulator

NeuroBoxingSimulator is a reproducible boxing strategy environment in which a
neural actor-critic learns tactics through masked PPO and self-play. Unlike the
original random-damage demo, every network output now affects a legal action in
the match.

## What it models

- Six simultaneous actions: jab, cross, hook, guard, dodge, and recover.
- Health, stamina, effective-strike scoring, KOs, and decisions.
- Power, speed, defense, and conditioning differences between fighters.
- State-dependent action masks, so exhausted fighters cannot select actions
  they cannot afford.
- Stamina-dependent accuracy and a bounded 4-10% punch-performance effect.
- PPO training against both a tactical baseline and a rotating pool of older
  neural policies.
- Seeded simulation, evaluation over alternating corners, model checkpoints,
  and a non-interactive match report plot.

The values are normalized game mechanics. This project is not a medical,
concussion, injury, or real-world punch-force predictor.

## Run

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
python main.py
```

Useful options:

```powershell
python main.py --episodes 640 --eval-matches 200 --seed 42
python main.py --load-model neuroboxer.pt --eval-matches 500
pytest -q
```

The default run writes `neuroboxer.pt` and `boxing_match.png`. Increase
`--episodes` for a stronger but slower training run.

## Design basis

The learning loop uses the clipped objective from
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), with
generalized advantage estimation, entropy regularization, gradient clipping,
and opponent snapshots. Self-play and opponent pools follow the practical
competitive-training pattern demonstrated by
[OpenAI Five](https://arxiv.org/abs/1912.06680).

Action masking is part of both training and inference. It is supported by
[Huang and Ontanon (2020)](https://arxiv.org/abs/2006.14171), while
[Zabounidis et al. (2026)](https://arxiv.org/abs/2603.09090) provides recent
evidence on why penalty-only handling can suppress actions that are valid in
other states.

The fatigue multiplier is deliberately conservative. A recent amateur-boxing
study reported an average post-fatigue punch-force decline of about 4.3%, with
larger declines around 10% for weaker punches:
[The Role of Lower Limb Kinetics in Boxing Punches and the Impact of Fatigue on
Biomechanical Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC12729554/).

## Structure

- `neuroboxing.py`: environment, agents, PPO trainer, evaluation, persistence,
  and plotting.
- `main.py`: command-line training and showcase workflow.
- `tests/test_neuroboxing.py`: rules, determinism, masking, persistence, and
  training smoke tests.
