"""Extract a fresh PPO buffer and one diagnostic update from a saved checkpoint.

Does not change checkpoints or claim to recover the original training buffer.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from skid_steer_rl import LoaderEnv


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.transitions = []

    def _on_step(self):
        data = self.locals
        self.transitions.append(
            {
                "next_observations": data["new_obs"].tolist(),
                "environment_rewards": data["rewards"].tolist(),
                "dones": data["dones"].tolist(),
                "timeouts": [
                    bool(i.get("TimeLimit.truncated", False)) for i in data["infos"]
                ],
                "terminal_observations": [
                    i.get("terminal_observation", np.array([])).tolist()
                    for i in data["infos"]
                ],
            }
        )
        return True


def extract():
    torch.set_num_threads(1)
    checkpoint = Path("runs/seed-7/final.zip")
    env = DummyVecEnv([lambda: LoaderEnv() for _ in range(4)])
    model = PPO.load(checkpoint, env=env, device="cpu")
    model.set_random_seed(2026)
    _, callback = model._setup_learn(2048, Capture(), reset_num_timesteps=False)
    callback.on_training_start(locals(), globals())
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=512)
    b = model.rollout_buffer
    with torch.no_grad():
        bootstrap = (
            model.policy.predict_values(torch.as_tensor(model._last_obs))
            .flatten()
            .tolist()
        )
    buffer = {
        k: getattr(b, k).tolist()
        for k in (
            "observations",
            "actions",
            "rewards",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
            "returns",
        )
    }
    # SB3 flattens environment-major; take one seeded shuffled minibatch.
    batch = next(b.get(256))
    model.policy.set_training_mode(True)
    advantages = (batch.advantages - batch.advantages.mean()) / (
        batch.advantages.std() + 1e-8
    )

    def outputs():
        values, logp, entropy = model.policy.evaluate_actions(
            batch.observations, batch.actions
        )
        dist = model.policy.get_distribution(batch.observations).distribution
        ratio = torch.exp(logp - batch.old_log_prob)
        policy = -torch.minimum(
            advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2)
        )
        value = (values.flatten() - batch.returns).square()
        total = policy.mean() + 0.5 * value.mean() - 0.005 * entropy.mean()
        return total, {
            "mean": dist.mean.detach().tolist(),
            "std": dist.stddev.detach().tolist(),
            "value": values.flatten().detach().tolist(),
            "log_prob": logp.detach().tolist(),
            "ratio": ratio.detach().tolist(),
            "policy_loss": policy.detach().tolist(),
            "value_loss": value.detach().tolist(),
            "entropy": entropy.detach().tolist(),
            "total_loss": total.item(),
        }

    loss, before = outputs()
    model.policy.optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.policy.parameters(), model.max_grad_norm
    )
    model.policy.optimizer.step()
    _, after = outputs()
    result = {
        "provenance": "Fresh diagnostic collection from seed-7 final checkpoint; not the historical training buffer. One additional Adam minibatch step in memory only.",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "collection_seed": 2026,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "buffer_shape": [512, 4],
        "minibatch_size": 256,
        "bootstrap_values": bootstrap,
        "last_dones": callback.transitions[-1]["dones"],
        "buffer": buffer,
        "transitions": callback.transitions,
        "minibatch": {
            "observations": batch.observations.tolist(),
            "actions": batch.actions.tolist(),
            "old_values": batch.old_values.tolist(),
            "old_log_prob": batch.old_log_prob.tolist(),
            "advantages": batch.advantages.tolist(),
            "normalized_advantages": advantages.tolist(),
            "returns": batch.returns.tolist(),
        },
        "before": before,
        "after": after,
        "gradient_norm_before_clipping": norm.item(),
    }
    Path("artifacts/training-trace.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    # Small browser payload: first minibatch row, preserving its buffer coordinates.
    obs = batch.observations[0].numpy()
    locations = np.argwhere(np.all(np.array(buffer["observations"]) == obs, axis=2))
    t, e = map(int, locations[0])
    row = {k: v[0] for k, v in result["minibatch"].items()}
    row.update(
        {
            "step": t,
            "environment": e,
            "reward": buffer["rewards"][t][e],
            "environment_reward": result["transitions"][t]["environment_rewards"][e],
            "next_observation": result["transitions"][t]["next_observations"][e],
            "done": result["transitions"][t]["dones"][e],
            "timeout": result["transitions"][t]["timeouts"][e],
        }
    )
    summary = {
        k: result[k]
        for k in (
            "provenance",
            "checkpoint_sha256",
            "collection_seed",
            "buffer_shape",
            "minibatch_size",
        )
    }
    summary["row"] = row
    for name in ("before", "after"):
        summary[name] = {
            k: (v[0] if isinstance(v, list) else v) for k, v in result[name].items()
        }
    Path("artifacts/training-example.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    env.close()


if __name__ == "__main__":
    extract()
