"""Two short PPO runs, independent training seeds, shared held-out evaluation seeds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from skid_steer_rl import LoaderEnv


def evaluate(model, seeds=range(1000, 1020)):
    env = LoaderEnv()
    records = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        for t in range(240):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        records.append({"seed": seed, "return": total, "steps": t + 1, **info})
    env.close()
    return records


class EvalLog(BaseCallback):
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.rows = []

    def _on_step(self):
        if self.num_timesteps % 16384 == 0:
            rows = evaluate(self.model, range(900, 905))
            record = {
                "steps": self.num_timesteps,
                "mean_return": float(np.mean([r["return"] for r in rows])),
                "successes": sum(r["success"] for r in rows),
                "episodes": len(rows),
            }
            self.rows.append(record)
            (self.out / "validation.json").write_text(json.dumps(self.rows, indent=2))
            print(record, flush=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=131072)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path("runs") / f"seed-{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(LoaderEnv()) for _ in range(4)])
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device="cpu",
        verbose=0,
        n_steps=512,
        batch_size=256,
        n_epochs=6,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.save(out / "initial")
    initial = evaluate(model)
    start = time.monotonic()
    model.learn(total_timesteps=args.steps, callback=EvalLog(out))
    elapsed = time.monotonic() - start
    model.save(out / "final")
    final = evaluate(model)
    result = {
        "training_seed": args.seed,
        "steps": model.num_timesteps,
        "training_seconds_including_validation": elapsed,
        "evaluation_policy": "deterministic",
        "initial": initial,
        "final": final,
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "seed": args.seed,
                "seconds": elapsed,
                "before": sum(r["success"] for r in initial),
                "after": sum(r["success"] for r in final),
            }
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
