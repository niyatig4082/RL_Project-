# DS-551 Final Project Setup

## 1) Create and activate the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run smoke test

```bash
python scripts/smoke_test.py
```

## 3) Run a PPO baseline training

```bash
python scripts/train_ppo_sb3.py
```

## 4) Run a DQN baseline training (SB3)

```bash
python scripts/train_dqn_sb3.py
```

## 5) Run a Double DQN baseline training (PyTorch)

```bash
python scripts/train_ddqn_torch.py
```

## 6) Evaluate trained models

```bash
# PPO
python scripts/evaluate_model.py --algo ppo --model-path outputs/ppo_seed_42.zip --episodes 30

# DQN
python scripts/evaluate_model.py --algo dqn --model-path outputs/dqn_seed_42.zip --episodes 30

# Double DQN
python scripts/evaluate_model.py --algo ddqn --model-path outputs/ddqn_seed_42.pt --episodes 30
```

## 7) Run multi-seed experiments (focus on code + results)

```bash
python scripts/run_multiseed.py \
  --algos ppo dqn ddqn \
  --seeds 0 1 2 \
  --ppo-steps 200000 \
  --dqn-steps 200000 \
  --ddqn-steps 200000 \
  --eval-freq 20000 \
  --eval-episodes 30
```

This writes aggregate evaluation metrics to `outputs/multiseed_summary.csv`.

## 8) Auto-plot comparison figures

```bash
python scripts/plot_results.py --algos ppo dqn ddqn --seeds 0 1 2
```

Plots are saved to `outputs/plots/`.

Model checkpoints are written to `outputs/` and tensorboard logs to `logs/`.

## 9) Export a simulation video (policy rollout)

```bash
# Example: record PPO rollout video
python scripts/evaluate_model.py \
  --algo ppo \
  --model-path outputs/ppo_seed_0.zip \
  --episodes 10 \
  --frame-stack 4 \
  --record-video \
  --video-dir outputs/videos \
  --video-prefix ppo_seed0_demo
```

Videos are saved under `outputs/videos/`.

## 10) Suggested tuning knobs (from project hints)

- Hyperparameters:
  - `train_ppo_sb3.py`: `--learning-rate`, `--n-steps`, `--batch-size`, `--ent-coef`
  - `train_dqn_sb3.py`: `--buffer-size`, `--batch-size`, `--exploration-fraction`, `--exploration-final-eps`
  - `train_ddqn_torch.py`: `--buffer-size`, `--batch-size`, `--epsilon-*`, `--target-update-freq`
- Partial observability:
  - Use `--frame-stack 4` for PPO/DQN and in evaluation/multiseed calls.
- Replay buffer variants:
  - DDQN now supports prioritized replay via `--prioritized-replay` (with `--per-alpha`, `--per-beta-start`, `--per-beta-end`).

## 11) Non-zero success recipe (recommended)

If you see `success_rate = 0`, use longer uninterrupted runs and tuned exploration/warmup:

```bash
python scripts/run_multiseed.py \
  --algos ppo dqn ddqn \
  --seeds 0 1 2 \
  --frame-stack 1 \
  --ppo-steps 120000 \
  --dqn-steps 180000 \
  --ddqn-steps 250000 \
  --eval-freq 10000 \
  --eval-episodes 30 \
  --dqn-learning-starts 2000 \
  --dqn-buffer-size 200000 \
  --dqn-batch-size 128 \
  --dqn-target-update-interval 2000 \
  --dqn-exploration-fraction 0.5 \
  --dqn-exploration-final-eps 0.02 \
  --ddqn-learning-starts 2000 \
  --ddqn-buffer-size 200000 \
  --ddqn-batch-size 128 \
  --ddqn-target-update-freq 2000 \
  --ddqn-epsilon-decay-fraction 0.5 \
  --ddqn-epsilon-end 0.02 \
  --ddqn-prioritized-replay
```

Important: avoid interrupting training early; sparse-reward FourRooms often needs longer horizons, especially for DDQN.
