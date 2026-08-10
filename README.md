# RL Project: DRL Indoor Navigation (MiniWorld FourRooms)

This project trains and compares three deep RL algorithms on MiniWorld FourRooms:

- PPO
- DQN
- DDQN

It supports training, evaluation, plotting, and simulation video generation.

## 1. Requirements

- OS: Windows (commands below use PowerShell)
- Python: 3.11 recommended
- GPU optional (CUDA supported through PyTorch)

Install external packages from [requirements.txt](requirements.txt).

## 2. Environment Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Main Pipeline (Train + Evaluate + Plot + Video)

Run the unified pipeline from [scripts/main.py](scripts/main.py):

```powershell
python scripts/main.py --allow-training --algos ppo dqn ddqn
```

This command will:

- train selected algorithms
- evaluate trained models
- save a summary CSV
- generate plots
- render videos

## 4. Common Commands

### 4.1 Train all 3 algorithms with custom steps

```powershell
python scripts/main.py --allow-training --algos ppo dqn ddqn --seed 42 --ppo-steps 500000 --dqn-steps 250000 --ddqn-steps 250000 --eval-freq 25000 --eval-episodes 20 --video-episodes 1 --device auto
```

### 4.2 Evaluate one model only

```powershell
python scripts/evaluate_model.py --algo ppo --model-path outputs/ppo_seed_42.zip --episodes 30 --seed 42 --device auto --summary-csv outputs/summary.csv
```

### 4.3 Render one video only (normal mode)

```powershell
python scripts/render_video.py --algo dqn --model-path outputs/dqn_seed_42.zip --episodes 1 --output-dir outputs --device auto
```

### 4.4 Render PPO success-only video

```powershell
python scripts/render_video.py --algo ppo --model-path outputs/ppo_seed_42.zip --output-dir outputs --device auto --until-success --episodes 1 --max-attempts 0
```

Notes:

- `--until-success` keeps trying episodes until a successful goal-reaching rollout is found.
- `--max-attempts 0` means no attempt limit.

## 5. Key Configuration Parameters

### 5.1 [scripts/main.py](scripts/main.py)

- `--allow-training`: required to start training pipeline
- `--algos`: list of algorithms (`ppo`, `dqn`, `ddqn`)
- `--seed`: base seed
- `--ppo-steps`, `--dqn-steps`, `--ddqn-steps`: training timesteps per algorithm
- `--eval-freq`: evaluation frequency during training
- `--eval-episodes`: number of eval episodes
- `--video-episodes`: episodes to render in standard mode
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`, etc.
- `--eval-only` with `--algo --model-path --episodes`: evaluate from [scripts/main.py](scripts/main.py) without training

### 5.2 Training scripts

- PPO: [scripts/train_ppo_sb3.py](scripts/train_ppo_sb3.py)
- DQN: [scripts/train_dqn_sb3.py](scripts/train_dqn_sb3.py)
- DDQN: [scripts/train_ddqn_sb3.py](scripts/train_ddqn_sb3.py)

Each script has additional hyperparameters via CLI (learning rate, buffer size, batch size, exploration schedule, etc.).

View all available options with:

```powershell
python scripts/train_ppo_sb3.py --help
python scripts/train_dqn_sb3.py --help
python scripts/train_ddqn_sb3.py --help
python scripts/main.py --help
python scripts/evaluate_model.py --help
python scripts/render_video.py --help
```

## 6. Output Structure

- Models: [outputs](outputs)
	- `ppo_seed_<seed>.zip`
	- `dqn_seed_<seed>.zip`
	- `ddqn_seed_<seed>.zip`
- Evaluation summary: [outputs/summary.csv](outputs/summary.csv)
- Plots: [outputs/plots](outputs/plots)
	- `reward_vs_timesteps.png`
	- `final_comparison.png`
- Videos: [outputs/videos](outputs/videos)

## 7. Environment/Reward Setup

- Environment: `MiniWorld-FourRooms-v0`
- Episode limit: 250 steps
- Reward wrapper: [scripts/reward_wrappers.py](scripts/reward_wrappers.py)
- Observation processing: image transpose + frame stacking (4 frames)

## 8. Troubleshooting

- If `python` is not recognized, run using virtualenv executable directly:

```powershell
.\.venv\Scripts\python.exe scripts/main.py --allow-training --algos ppo dqn ddqn
```

- If CUDA is unavailable, set `--device cpu`.

- If video rendering warns about frame resize to `(800, 608)`, this is expected codec compatibility behavior.
