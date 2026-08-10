# RL Project: Clean Start Training Guide

This project trains PPO, DQN, and DDQN on `MiniWorld-FourRooms-v0` using a single streamlined entry point.

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

## 3) Run the full workflow

```powershell
python scripts/main.py --seed 42 --algos ppo ddqn --ppo-steps 200000 --ddqn-steps 200000 --eval-freq 20000 --eval-episodes 30 --allow-training
```

This single command will:
- train PPO and DDQN,
- evaluate both models,
- generate summary plots,
- and validate the required outputs.

Saved models go to `outputs/` and eval logs go to `logs/`.

## 4) Core scripts

- `scripts/train_ppo_sb3.py`
- `scripts/train_dqn_sb3.py`
- `scripts/train_ddqn_sb3.py`
- `scripts/reward_shaping.py`
- `scripts/main.py`
