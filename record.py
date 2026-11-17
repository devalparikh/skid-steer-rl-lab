"""Record predetermined evaluation episodes, with exact states/actions and file hashes."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from skid_steer_rl import LoaderEnv


def main():
    torch.set_num_threads(1)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    manifest = []
    # Chosen before viewing results, to avoid choosing only successful videos.
    for training_seed, checkpoint in [(7, "initial"), (7, "final"), (19, "final")]:
        model = PPO.load(f"runs/seed-{training_seed}/{checkpoint}.zip", device="cpu")
        env = LoaderEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=1000)
        name = f"seed-{training_seed}-{checkpoint}-episode-1000"
        rows = []
        total = 0
        writer = imageio.get_writer(
            out / f"{name}.mp4", fps=20, codec="libx264", quality=7
        )
        try:
            for t in range(240):
                action, _ = model.predict(obs, deterministic=True)
                old_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                rows.append(
                    {
                        "step": t + 1,
                        "observation": old_obs.tolist(),
                        "action": action.tolist(),
                        "reward": reward,
                        "qpos": env.data.qpos.tolist(),
                        "qvel": env.data.qvel.tolist(),
                        "terminated": terminated,
                        "truncated": truncated,
                        **info,
                    }
                )
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 720, 54), fill=(24, 29, 33))
                draw.text(
                    (16, 10),
                    f"SIMULATION | seed {training_seed} | {checkpoint} policy | evaluation 1000",
                    fill="white",
                )
                draw.text(
                    (16, 32),
                    f"t={info['time']:.2f}s  distance={info['distance']:.2f}m  speed={info['speed']:.2f}m/s  return={total:.2f}",
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
                if t == 0:
                    frame.save(out / f"{name}.png")
                if terminated or truncated:
                    break
        finally:
            writer.close()
            env.close()
        payload = {
            "training_seed": training_seed,
            "checkpoint": checkpoint,
            "evaluation_seed": 1000,
            "goal": env.goal.tolist(),
            "deterministic": True,
            "frames_per_second": 20,
            "transitions": rows,
        }
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
        manifest.append(
            {
                "name": name,
                "return": total,
                "steps": len(rows),
                "final": rows[-1],
                "sha256": {
                    ext: hashlib.sha256(
                        (out / f"{name}.{ext}").read_bytes()
                    ).hexdigest()
                    for ext in ["mp4", "json", "png"]
                },
            }
        )
        print(name, len(rows), info, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
