# DS-551 Final Project - Deep RL on MiniWorld FourRooms

Worcester Polytechnic Institute - Summer 2026

This project trains and compares three DRL methods on MiniWorld-FourRooms-v0:

- PPO
- DQN
- DDQN baseline

All training implementations use Stable-Baselines3 only. The DDQN-labeled run is implemented as an SB3 DQN baseline (no custom DDQN class), so the comparison remains consistent at the framework level.

## Project Layout

- `scripts/` - training, evaluation, plotting, and rendering scripts
- `outputs/` - trained models, plots, videos, summary CSV, and PPT text extraction artifacts
- `logs/` - TensorBoard/evaluation logs and checkpoints
- `requirements.txt` - package versions used in this repository

## Requirements

Python 3.11 recommended (Windows PowerShell commands shown below).

Install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Key versions in `requirements.txt`:

- miniworld 2.1.0
- stable-baselines3 2.9.0
- gymnasium 1.3.0
- torch 2.13.0+cu126

## Environment and Reward Setup

- Environment: `MiniWorld-FourRooms-v0`
- Observation: first-person RGB, transposed + frame stacking (`n_stack=4`)
- Episode horizon: 250 steps
- Reward wrapper: `scripts/reward_wrappers.py` (`StrictGoalRewardWrapper`)
- Rendering textures: MiniWorld default packaged textures

## Main Scripts

| Script | Purpose | Typical Output |
|---|---|---|
| `scripts/train_ppo_sb3.py` | Train PPO | `outputs/ppo_<run_name>.zip`, `logs/ppo/<run_name>/...` |
| `scripts/train_dqn_sb3.py` | Train DQN | `outputs/dqn_<run_name>.zip`, `logs/dqn/<run_name>/...` |
| `scripts/train_ddqn_sb3.py` | Train DDQN-labeled baseline | `outputs/ddqn_<run_name>.zip`, `logs/ddqn/<run_name>/...` |
| `scripts/evaluate_model.py` | Evaluate one trained model | `outputs/summary.csv` row append |
| `scripts/plot_results.py` | Generate training/eval comparison plots | `outputs/plots/*.png` |
| `scripts/render_video.py` | Render policy rollout videos | `outputs/videos/*.mp4` |
| `scripts/main.py` | Router to train/eval/plot/video pipeline | all of the above |

## How To Run

### Option A: Full pipeline from router

```powershell
python scripts/main.py --allow-training --algos ppo dqn ddqn --seed 42 --ppo-steps 1000000 --dqn-steps 1000000 --ddqn-steps 1000000 --eval-freq 25000 --eval-episodes 30 --video-episodes 1 --device auto
```

### Option B: Run scripts individually (final-style)

```powershell
python scripts/train_ppo_sb3.py --allow-training --seed 42 --total-timesteps 1000000 --n-envs 8 --eval-freq 25000 --eval-episodes 20 --checkpoint-total-steps 100000 --run-name ppo_refstyle_v2 --device auto

python scripts/train_dqn_sb3.py --allow-training --seed 42 --total-timesteps 1000000 --eval-freq 25000 --eval-episodes 20 --checkpoint-total-steps 250000 --run-name dqn_refstyle_v2 --device auto

python scripts/train_ddqn_sb3.py --allow-training --seed 42 --total-timesteps 1000000 --eval-freq 25000 --eval-episodes 20 --checkpoint-total-steps 250000 --run-name ddqn_refstyle_v2 --device auto
```

Evaluate saved models and update summary:

```powershell
python scripts/evaluate_model.py --algo ppo --model-path outputs/ppo_ppo_refstyle_v2.zip --episodes 30 --seed 42 --device auto --summary-csv outputs/summary.csv
python scripts/evaluate_model.py --algo dqn --model-path outputs/dqn_dqn_refstyle_v2.zip --episodes 30 --seed 42 --device auto --summary-csv outputs/summary.csv
python scripts/evaluate_model.py --algo ddqn --model-path outputs/ddqn_ddqn_refstyle_v2.zip --episodes 30 --seed 42 --device auto --summary-csv outputs/summary.csv
```

Generate plots:

```powershell
python scripts/plot_results.py --summary-csv outputs/summary.csv --output-dir outputs --log-dir logs
```

Render videos:

```powershell
python scripts/render_video.py --algo ppo --model-path outputs/ppo_ppo_refstyle_v2.zip --output-dir outputs --device auto --until-success --episodes 1 --max-attempts 0
python scripts/render_video.py --algo dqn --model-path outputs/dqn_dqn_refstyle_v2.zip --output-dir outputs --device auto --episodes 1
python scripts/render_video.py --algo ddqn --model-path outputs/ddqn_ddqn_refstyle_v2.zip --output-dir outputs --device auto --episodes 1
```

## Final Hyperparameters Used (Reported)

### PPO (`scripts/train_ppo_sb3.py`)

- total timesteps: 1,000,000
- n_envs: 8
- learning rate: 2.5e-4
- n_steps: 256
- batch size: 512
- n_epochs: 4
- gamma: 0.99
- gae_lambda: 0.97
- clip_range: 0.2
- ent_coef: 0.01
- eval freq: 25,000 (callback frequency)
- checkpoint interval target: 100,000 global steps

### DQN / DDQN baseline (`scripts/train_dqn_sb3.py`, `scripts/train_ddqn_sb3.py`)

- total timesteps: 1,000,000
- learning rate: 1e-4
- buffer size: 100,000
- learning starts: 10,000
- batch size: 32
- gamma: 0.99
- train freq: 4
- gradient steps: 1
- target update interval: 1,000
- exploration: epsilon 1.0 -> 0.05 across 30% of training
- eval freq: 25,000
- checkpoint interval target: 250,000 global steps

## Final Results (From `outputs/summary.csv`)

Deterministic evaluation, seed 42, 30 episodes each.

| Algorithm | Mean Return | Success Rate | Mean Steps |
|---|---:|---:|---:|
| PPO | 0.3544 | 36.67% | 173.63 |
| DQN | 0.0331 | 3.33% | 241.93 |
| DDQN baseline | 0.0000 | 0.00% | 250.00 |

Interpretation: PPO clearly outperformed both value-based baselines on sparse-reward FourRooms under this training budget.

## Current Output Artifacts

### Models

- `outputs/ppo_ppo_refstyle_v2.zip`
- `outputs/dqn_dqn_refstyle_v2.zip`
- `outputs/ddqn_ddqn_refstyle_v2.zip`

### Plots

- `outputs/plots/ppo_training_curve.png`
- `outputs/plots/algorithm_comparison.png`
- `outputs/plots/reward_vs_timesteps.png`

### Videos

- `outputs/videos/ppo_ppo_ppo_refstyle_v2.mp4`
- `outputs/videos/dqn_dqn_dqn_refstyle_v2.mp4`
- `outputs/videos/ddqn_ddqn_ddqn_refstyle_v2.mp4`

## Notes

- `import miniworld` may look unused, but it is required to register MiniWorld environments with Gymnasium.
- If CUDA is unavailable, use `--device cpu`.
- For reproducibility, keep seed and run-name conventions consistent across training/evaluation/rendering commands.
