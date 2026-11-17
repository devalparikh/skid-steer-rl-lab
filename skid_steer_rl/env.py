"""MuJoCo skid-steer parking task. No calibrated hydraulics or hardware interface."""

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class LoaderEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).with_name("loader.xml"))
        )
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.renderer = None
        self.goal = np.zeros(2)
        self.previous = np.zeros(2)
        self.ticks = 0
        self.held = 0

    def _obs(self):
        rotation = self.data.xmat[1].reshape(3, 3)
        local = rotation[:2, :2].T @ (self.goal - self.data.qpos[:2])
        velocity = rotation.T @ self.data.qvel[:3]
        return np.array(
            [
                *(local / 4),
                velocity[0] / 3,
                velocity[1] / 3,
                self.data.qvel[5] / 3,
                rotation[2, 2],
                *self.previous,
                self.held / 10,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        angle = self.np_random.uniform(-1.2, 1.2)
        distance = self.np_random.uniform(2.5, 4)
        self.goal = distance * np.array([np.cos(angle), np.sin(angle)])
        self.model.site_pos[0, :2] = self.goal
        self.previous = np.zeros(2)
        self.ticks = self.held = 0
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)
        self.distance = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        self.data.ctrl[:4] = 7 * action[[0, 1, 0, 1]]
        # Arm and bucket are physically articulated but held by fixed servos.
        self.data.ctrl[4:] = 0
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.ticks += 1
        distance = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        speed = float(np.linalg.norm(self.data.qvel[:2]))
        self.held = self.held + 1 if distance < 0.35 and speed < 0.25 else 0
        success = self.held >= 10
        tipped = self.data.xmat[1].reshape(3, 3)[2, 2] < 0.7
        escaped = distance > 10
        reward = (
            5 * (self.distance - distance)
            - 0.01
            - 0.002 * float(np.square(action).sum())
        )
        reward += 0.1 if distance < 0.35 and speed < 0.25 else 0
        reward += 10 if success else 0
        reward -= 10 if tipped or escaped else 0
        self.previous = action.copy()
        self.distance = distance
        return (
            self._obs(),
            float(reward),
            bool(success or tipped or escaped),
            self.ticks >= 240,
            {
                "success": bool(success),
                "distance": distance,
                "speed": speed,
                "tipped": bool(tipped),
                "time": self.ticks * 0.05,
            },
        )

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=720)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.5, 0, 0.5]
        camera.distance = 9
        camera.azimuth = 135
        camera.elevation = -28
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
