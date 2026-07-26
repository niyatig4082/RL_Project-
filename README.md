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
python scripts/evaluate_model.py --algo ppo --model-path outputs/ppo_fourrooms.zip --episodes 30

# DQN
python scripts/evaluate_model.py --algo dqn --model-path outputs/dqn_fourrooms.zip --episodes 30

# Double DQN
python scripts/evaluate_model.py --algo ddqn --model-path outputs/ddqn_fourrooms.pt --episodes 30
```

Model checkpoints are written to `outputs/` and tensorboard logs to `logs/`.
