# RL Project Results Report

## Overview
This report summarizes the current experiment outcomes for the MiniWorld FourRooms RL project using the PPO, DQN, and DDQN training pipelines.

## Environment and setup
- Environment: MiniWorld FourRooms
- Episode horizon: 250 steps
- Reward rule: sparse goal reward, zero on normal steps, positive only when the agent reaches the red box
- Observation handling: frame stacking used for partial observability

## Verified results
The values below were extracted from the saved evaluation checkpoint files in the project logs directory and the latest summary CSV.

### PPO
- Best mean evaluation reward: 0.57936
- Final mean evaluation reward: 0.39008
- Best checkpoint timestep: 275,000

### DQN
- Best mean evaluation reward: 0.09992
- Final mean evaluation reward: 0.0
- Best checkpoint timestep: 75,000

### DDQN
- Best mean evaluation reward: 0.09992
- Final mean evaluation reward: 0.0
- Best checkpoint timestep: 75,000

## Summary CSV snapshot
The latest saved summary row is:

- algo: ddqn
- seed: 42
- mean_return: 0.0
- success_rate: 0.0
- mean_steps: 250.0

## Interpretation
- PPO clearly outperformed DQN and DDQN in the current runs.
- DQN and DDQN were not able to learn a reliable successful policy under the sparse reward setting within the current training budget.
- The assignment requirement of a 250-step episode horizon is implemented in the environment and wrapper setup.

## Files generated
- Trained models:
  - outputs/ppo_seed_42.zip
  - outputs/dqn_seed_42.zip
  - outputs/ddqn_seed_42.zip
- Plots:
  - outputs/plots/reward_vs_timesteps.png
  - outputs/plots/final_comparison.png
- Videos:
  - outputs/videos/ppo_ppo_seed_42.mp4
  - outputs/videos/dqn_dqn_seed_42.mp4

## Notes
- These are the results currently available on disk from the latest completed evaluation checkpoints.
- If more training runs are executed later, this report should be refreshed with the new metrics.
