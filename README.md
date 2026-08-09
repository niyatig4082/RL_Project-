# RL Project: Clean Start Training Guide

This project trains PPO and DQN on `MiniWorld-FourRooms-v0` using a single streamlined entry point.

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
python scripts/main.py --seeds 42 --ppo-steps 200000 --dqn-steps 200000 --eval-freq 20000 --eval-episodes 30
```

This single command will:
- train PPO and DQN,
- evaluate both models,
- generate summary plots,
- and validate the required outputs.

Saved models go to `outputs/` and eval logs go to `logs/`.

## 4) Core scripts

- `scripts/train_ppo_sb3.py`
- `scripts/train_dqn_sb3.py`
- `scripts/reward_shaping.py`
- `scripts/main.py`
