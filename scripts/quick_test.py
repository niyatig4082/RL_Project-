from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("[test]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable

    run([py, "scripts/smoke_test.py"])
    run([py, "scripts/inspect_env.py", "--steps", "3", "--save-dir", "outputs/inspect_quick_test"])
    run([py, "scripts/train_ppo_sb3.py", "--help"])
    run([py, "scripts/train_dqn_sb3.py", "--help"])
    run([py, "scripts/train_ddqn_torch.py", "--help"])
    run([py, "scripts/evaluate_model.py", "--help"])
    run([py, "scripts/run_multiseed.py", "--help"])
    run([py, "scripts/plot_results.py", "--help"])
    run([py, "scripts/validate_outputs.py", "--help"])

    print("quick tests passed")


if __name__ == "__main__":
    main()
