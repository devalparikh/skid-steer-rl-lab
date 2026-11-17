# Skid-steer RL lab

A real, deliberately small reinforcement-learning project: a procedural 3D skid-steer loader in MuJoCo, a Gymnasium parking environment, and two short CPU PPO runs. The model has a free chassis, four driven wheels, ground contacts, and articulated lift/bucket joints held by fixed servos. The policy steers and drives; it does not control the bucket.

Inspired by [Heavy Metal Acrobatics](https://arxiv.org/abs/2611.18365), which controls an actual John Deere 332G. **This is an original simplified model, not manufacturer CAD, the paper's simulator, or a reproduction of its wheelies.** No external meshes, footage, textures, or weights are redistributed. There is no hardware-control interface.

## Run

Python 3.12 on Linux was used for the checked-in runs. Install CPU PyTorch first to avoid CUDA downloads:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m pytest -q
python train.py --seed 7 --steps 131072
python train.py --seed 19 --steps 131072
MUJOCO_GL=egl python record.py
```

`requirements-lock.txt` records the exact installed dependencies, including test/format tools. CPU PyTorch's wheel requires the index above; the lockfile alone does not configure that index. Rendering requires working EGL/OpenGL. On a desktop, `MUJOCO_GL=glfw` is another option; software rendering via `osmesa` needs the OS Mesa libraries. Training needs no renderer or GPU.

## Task contract

- Reset at rest on flat ground. Sample a target 2.5–4 m away, at a bearing within ±1.2 radians.
- Observe body-frame goal offset, forward/lateral velocity, yaw rate, uprightness, previous left/right commands, and the parking hold counter (9 scalars).
- Output left and right commands in [-1, 1]. Each drives the front and rear wheel on that side; velocity actuator setpoints are scaled to ±7 rad/s.
- MuJoCo timestep: 0.005 seconds, 10 physics steps per policy action (20 Hz control).
- Reward: `5 * distance_progress - 0.01 - 0.002 * sum(action²)`, plus 0.1 while within 0.35 m at less than 0.25 m/s, plus 10 on success. Tipping or moving over 10 m from the goal costs 10 and terminates.
- Success requires staying within those position/speed limits for 10 consecutive actions (0.5 seconds).
- Timeout after 240 actions (12 seconds) is **truncation**, not terminal failure. SB3 handles time-limit bootstrapping through its vector environment wrapper.

The learning task uses simulator state, not vision. Fixed servos hold the arm; there is no digging, soil, load, hydraulic delay, sensor noise, terrain randomization, or balance task. Masses and actuator settings are teaching choices, not identified machine parameters. Collision geometry is simple primitives. These limits make the project feasible on a 4-vCPU/8-GB VM and rule out hardware claims.

## Experiments and artifacts

Each run uses PPO with four sequential vector environments, a two-layer 64-unit actor/critic, 512 rollout steps per environment, batch size 256, six update epochs, learning rate 0.0003, discount 0.99, GAE lambda 0.95, and entropy coefficient 0.005. Each run collects 131,072 transitions; checkpoints are saved before and after training.

Training seeds are 7 and 19. Intermediate checks use seeds 900–904. Final evaluation uses 20 separate seeds, 1000–1019, with deterministic actions before and after training. No checkpoint is selected by final test performance. Videos always show seed 1000, chosen in `record.py` before inspecting the evaluation outcomes. They are examples, not the aggregate evaluation.

- `runs/seed-*/evaluation.json`: all initial/final episode metrics and elapsed training time.
- `runs/seed-*/validation.json`: intermediate validation measurements, not training reward.
- `runs/seed-*/*.zip`: SB3 initial and final checkpoints (load trusted checkpoints only).
- `artifacts/*.mp4`: real simulator rollouts at 20 fps, one frame per control step, no time compression.
- `artifacts/*.json`: corresponding actions, rewards, complete qpos/qvel states, outcomes and hashes.
- `RESULTS.md`: aggregate results and interpretation.

Re-running the commands overwrites the corresponding seed's outputs. Preserve prior runs first if comparing changes. Exact bitwise reproduction may vary with platform and dependency versions.

## Why this stack?

[MuJoCo's MJCF](https://mujoco.readthedocs.io/en/stable/modeling.html) lets us specify bodies, joints, inertia, contacts, and actuators explicitly. For this experiment, inspectable primitive geometry is more useful than an image-generated mesh with unknown collision shapes and mass properties. A richer visual mesh could be attached later without replacing collision and inertia modeling.

[Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) supplies the reset/step contract. [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) supplies a tested trainer and supports CPU training with a small MLP. This project uses the public library rather than an improvised PPO implementation. It does not require a GPU robotics platform or a proprietary simulator.

## License

Original code, procedural MJCF model and generated artifacts: MIT, see LICENSE. Dependencies retain their own licenses: MuJoCo (Apache-2.0), Gymnasium (MIT), Stable-Baselines3 (MIT), PyTorch (BSD-style). Research links and names identify inspiration only; no endorsement or affiliation is implied.
