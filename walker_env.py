import numpy as np
import gymnasium as gym
import mujoco


def torso_id(model):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if body_id < 0:
        raise RuntimeError("Could not find the torso body.")
    return body_id


class FixedTorsoMass(gym.Wrapper):
    def __init__(self, env, torso_mass_scale=1.0):
        super().__init__(env)

        scale = float(torso_mass_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("torso_mass_scale must be finite and positive.")

        raw = env.unwrapped
        model = raw.model
        body_id = torso_id(model)

        self.torso_mass_scale = scale
        self.original_torso_mass = float(model.body_mass[body_id])
        self.original_total_mass = float(model.body_mass.sum())

        if scale != 1.0:
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale
            mujoco.mj_setConst(model, raw.data)

        self.current_torso_mass = float(model.body_mass[body_id])
        self.current_total_mass = float(model.body_mass.sum())


class RandomTorsoMass(gym.Wrapper):
    def __init__(self, env, low=1.0, high=2.0, seed=0):
        super().__init__(env)
        if not 0 < low <= high:
            raise ValueError("Need 0 < low <= high.")
        self.low, self.high = float(low), float(high)
        self.rng = np.random.default_rng(seed)
        model = env.unwrapped.model
        self.torso_id = torso_id(model)
        self.base_mass = float(model.body_mass[self.torso_id])
        self.base_inertia = model.body_inertia[self.torso_id].copy()
        self.scale = 1.0

    def reset(self, **kwargs):
        model, data = self.env.unwrapped.model, self.env.unwrapped.data
        self.scale = float(self.rng.uniform(self.low, self.high))
        model.body_mass[self.torso_id] = self.base_mass * self.scale
        model.body_inertia[self.torso_id] = self.base_inertia * self.scale
        mujoco.mj_setConst(model, data)
        return self.env.reset(**kwargs)


def make_env(torso_mass_scale=1.0, render_mode=None):
    base = gym.make(
        "Walker2d-v5",
        render_mode=render_mode,
        width=800,
        height=480,
        max_episode_steps=1000,
    )
    try:
        return FixedTorsoMass(base, torso_mass_scale)
    except Exception:
        base.close()
        raise
