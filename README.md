# RL Project: Clean Start Training Guide

This project trains PPO, DQN, and Double DQN on `MiniWorld-FourRooms-v0`.

## 1) Environment setup

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Preflight check

```powershell
python scripts/smoke_test.py
python scripts/quick_test.py
```

## 3) Start training (single run)

PPO:

```powershell
python scripts/train_ppo_sb3.py --seed 42 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 30
```

DQN:

```powershell
python scripts/train_dqn_sb3.py --seed 42 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 30
```

DDQN:

```powershell
python scripts/train_ddqn_torch.py --seed 42 --total-steps 200000 --eval-every 20000 --eval-episodes 30
```

Saved models go to `outputs/` and eval logs go to `logs/`.

## 4) Evaluate a trained model

```powershell
python scripts/evaluate_model.py --algo ppo --model-path outputs/ppo_seed_42.zip --episodes 30
python scripts/evaluate_model.py --algo dqn --model-path outputs/dqn_seed_42.zip --episodes 30
python scripts/evaluate_model.py --algo ddqn --model-path outputs/ddqn_seed_42.pt --episodes 30
```

## 5) Multi-seed benchmark + plots

```powershell
python scripts/run_multiseed.py --algos ppo dqn ddqn --seeds 0 1 2 --ppo-steps 200000 --dqn-steps 200000 --ddqn-steps 200000 --eval-freq 20000 --eval-episodes 30
python scripts/plot_results.py --algos ppo dqn ddqn --seeds 0 1 2
python scripts/validate_outputs.py
```

## Core scripts

- `scripts/train_ppo_sb3.py`
- `scripts/train_dqn_sb3.py`
- `scripts/train_ddqn_torch.py`
- `scripts/reward_shaping.py`
- `scripts/evaluate_model.py`
- `scripts/run_multiseed.py`
- `scripts/plot_results.py`
- `scripts/validate_outputs.py`
