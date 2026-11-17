import hashlib
import json
from pathlib import Path

import numpy as np

from skid_steer_rl import LoaderEnv

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_actions_reproduce_states_rewards_and_outcomes():
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    for item in manifest:
        for extension, expected in item["sha256"].items():
            path = ROOT / "artifacts" / f"{item['name']}.{extension}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        payload = json.loads((ROOT / "artifacts" / f"{item['name']}.json").read_text())
        env = LoaderEnv()
        observation, _ = env.reset(seed=payload["evaluation_seed"])
        for row in payload["transitions"]:
            np.testing.assert_allclose(observation, row["observation"], atol=1e-6)
            observation, reward, terminated, truncated, info = env.step(row["action"])
            np.testing.assert_allclose(env.data.qpos, row["qpos"], atol=1e-6)
            assert abs(reward - row["reward"]) < 1e-6
            assert terminated == row["terminated"]
            assert truncated == row["truncated"]
            assert info["success"] == row["success"]
        env.close()
