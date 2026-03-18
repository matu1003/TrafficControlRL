"""
envs_chain.py
=============
Extension of SingleLaneEnv for multi-vehicle chain training.

Components
----------
SingleLaneEnv           – unchanged physics core (imported from envs.py)
BasisActionWrapper      – unchanged (imported from envs.py)
PolyLeaderEnv           – leader acceleration = random degree-≤3 polynomial,
                          resampled each episode.
DatasetLeaderEnv        – leader acceleration sampled at O(1) cost from a
                          pre-generated .npy dataset of shape (N, n_total).
                          No live rollouts during training.

Dataset generation
------------------
generate_leader_dataset()  – roll out a frozen PPO model N times (or sample N
                              polynomials) and save leading_al profiles to disk.

Typical usage
-------------
  # Before training pair k+1:
  generate_leader_dataset(model, vec_norm, env_kwargs, n=500,
                          save_path="chain/pair0_dataset.npy")

  # Env factory for pair k+1:
  env = DatasetLeaderEnv(dataset_path="chain/pair0_dataset.npy", **env_kwargs)
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import SingleLaneEnv, BasisActionWrapper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _poly_acceleration(rng: np.random.Generator,
                        n_total: int,
                        a_min: float,
                        a_max: float,
                        degree: int = 3) -> np.ndarray:
    """
    Sample a random polynomial of degree ≤ `degree`, evaluated over
    [0, 1] with n_total points, rescaled to [a_min, a_max].
    Returns shape (n_total,) float32.
    """
    t = np.linspace(0.0, 1.0, n_total)
    coeffs = rng.uniform(-1.0, 1.0, size=degree + 1)
    a_raw = np.polyval(coeffs, t)
    lo, hi = a_raw.min(), a_raw.max()
    if hi > lo:
        a_raw = (a_raw - lo) / (hi - lo)          # → [0, 1]
        a_raw = a_raw * (a_max - a_min) + a_min   # → [a_min, a_max]
    else:
        a_raw = np.zeros(n_total)
    return a_raw.astype(np.float32)


def _single_rollout(model: PPO,
                    vec_norm: VecNormalize,
                    env_kwargs: dict,
                    K: int,
                    seed: int) -> np.ndarray:
    """
    Roll out `model` once (deterministic) and return the ego acceleration
    profile a(t) of shape (n_total,).  The env is created fresh each call
    so parallel seeds never interfere.
    """
    def _make():
        e = SingleLaneEnv(**env_kwargs)
        return BasisActionWrapper(e, K=K)

    rollout_env = DummyVecEnv([_make])
    rollout_env = VecNormalize(rollout_env,
                               norm_obs=True, norm_reward=False,
                               clip_obs=10.0, training=False)
    rollout_env.obs_rms = vec_norm.obs_rms
    rollout_env.ret_rms = vec_norm.ret_rms

    obs = rollout_env.reset()   # seed ignored here; diversity comes from
                                # the stochastic leader inside the env
    done = False
    base_env: SingleLaneEnv = rollout_env.envs[0].env  # Basis → Single

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = rollout_env.step(action)
        done = bool(dones[0])

    a_profile = base_env.state["traj"][4, :].copy()
    rollout_env.close()
    return a_profile.astype(np.float32)


# ---------------------------------------------------------------------------
# Public: dataset generation
# ---------------------------------------------------------------------------

def generate_leader_dataset(save_path: str,
                             n: int,
                             env_kwargs: dict,
                             model: PPO = None,
                             vec_norm: VecNormalize = None,
                             K: int = 20,
                             poly_degree: int = 3,
                             base_seed: int = 0) -> np.ndarray:
    """
    Generate a dataset of `n` leader acceleration profiles and save to disk.

    Two modes
    ---------
    Polynomial mode  (model=None):
        Sample n random degree-≤poly_degree polynomials.  Fast, no model needed.
        Used for pair 0 (vehicle 0 → vehicle 1).

    Rollout mode  (model provided):
        Run `model` deterministically n times on its own training env
        (which has a stochastic leader internally, so each episode differs).
        Used for pairs 1 and 2.

    Parameters
    ----------
    save_path   : str           Path for the output .npy file.
    n           : int           Number of profiles to generate.
    env_kwargs  : dict          Keyword args for SingleLaneEnv (needed for
                                n_total, a_min, a_max, and rollout env).
    model       : PPO | None    Frozen model to roll out (rollout mode).
    vec_norm    : VecNormalize  Norm stats matching the frozen model.
    K           : int           Number of basis coefficients (rollout mode).
    poly_degree : int           Max polynomial degree (polynomial mode).
    base_seed   : int           RNG seed offset.

    Returns
    -------
    dataset : np.ndarray  shape (n, n_total), float32
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Derive n_total from env_kwargs
    T        = env_kwargs.get("T", 40.0)
    dt       = env_kwargs.get("dt", 0.05)
    n_total  = int(round(T / dt))
    a_min    = env_kwargs.get("a_min", -9.0)
    a_max    = env_kwargs.get("a_max",  4.0)

    dataset = np.zeros((n, n_total), dtype=np.float32)

    if model is None:
        # ---- polynomial mode ----
        print(f"Generating {n} polynomial leader profiles → {save_path}")
        rng = np.random.default_rng(base_seed)
        for i in range(n):
            dataset[i] = _poly_acceleration(rng, n_total, a_min, a_max,
                                            degree=poly_degree)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n}")
    else:
        # ---- rollout mode ----
        print(f"Generating {n} rollout leader profiles → {save_path}")
        for i in range(n):
            dataset[i] = _single_rollout(model, vec_norm, env_kwargs,
                                         K=K, seed=base_seed + i)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n}")

    np.save(save_path, dataset)
    print(f"✅  Dataset saved: {save_path}  shape={dataset.shape}")
    return dataset


# ---------------------------------------------------------------------------
# PolyLeaderEnv  –  Vehicle 0 leader (infinite, on-the-fly polynomials)
# ---------------------------------------------------------------------------

class PolyLeaderEnv(SingleLaneEnv):
    """
    SingleLaneEnv with a fresh random polynomial leader at every reset().
    Used as the base environment for pair 0.

    Parameters
    ----------
    poly_degree : int   Maximum polynomial degree (default 3).
    All other kwargs forwarded to SingleLaneEnv.
    """

    def __init__(self, poly_degree: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.poly_degree = poly_degree
        self.leading_al = np.zeros(self.n_total, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # gym.Env.reset sets self.np_random
        super().reset(seed=seed, options=options)

        # New polynomial each episode
        self.leading_al = _poly_acceleration(
            self.np_random,
            n_total=self.n_total,
            a_min=self.a_min,
            a_max=self.a_max,
            degree=self.poly_degree,
        )

        # Re-integrate with the new leader profile
        traj = np.zeros((5, self.n_total), dtype=np.float32)
        traj[:, 0] = np.array(
            [self.xl0, self.vl0, self.x0, self.v0, 0], dtype=np.float32
        )
        X = traj[:, 0].astype(np.float64, copy=True)
        for i in range(self.n_total - 1):
            u = 0.0
            k1 = self._f(X, u, i)
            k2 = self._f(X + 0.5 * self.dt * k1, u, i)
            k3 = self._f(X + 0.5 * self.dt * k2, u, i)
            k4 = self._f(X + self.dt * k3, u, i)
            X = X + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            X[4] = k1[3]
            traj[:, i + 1] = X.astype(np.float32)

        self.state = {"traj": traj, "t": 0.0}
        self.control = np.zeros(self.n_total)
        return self._get_obs(), {}


# ---------------------------------------------------------------------------
# DatasetLeaderEnv  –  O(1) reset, no live rollouts
# ---------------------------------------------------------------------------

class DatasetLeaderEnv(SingleLaneEnv):
    """
    SingleLaneEnv whose leader acceleration is sampled at O(1) cost from a
    pre-generated dataset of profiles stored on disk.

    At reset() a random row is drawn from the dataset array (already loaded
    into RAM) and used as `leading_al`.  The leader trajectory is then
    re-integrated from initial conditions — the only work done at reset time.

    Parameters
    ----------
    dataset_path : str
        Path to a .npy file of shape (N, n_total) produced by
        generate_leader_dataset().
    All other kwargs forwarded to SingleLaneEnv.
    """

    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(**kwargs)

        dataset = np.load(dataset_path)          # shape (N, n_total)
        assert dataset.ndim == 2, "Dataset must be 2-D (N, n_total)"
        assert dataset.shape[1] == self.n_total, (
            f"Dataset n_total={dataset.shape[1]} != env n_total={self.n_total}"
        )
        self._dataset = dataset.astype(np.float32)
        self._n_profiles = dataset.shape[0]
        self.leading_al = np.zeros(self.n_total, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # gym.Env.reset sets self.np_random (used for index sampling)
        gym.Env.reset(self, seed=seed)

        # O(1): pick a random profile from the pre-generated dataset
        idx = int(self.np_random.integers(0, self._n_profiles))
        self.leading_al = self._dataset[idx].copy()

        # Re-integrate with the selected leader profile
        traj = np.zeros((5, self.n_total), dtype=np.float32)
        traj[:, 0] = np.array(
            [self.xl0, self.vl0, self.x0, self.v0, 0], dtype=np.float32
        )
        X = traj[:, 0].astype(np.float64, copy=True)
        for i in range(self.n_total - 1):
            u = 0.0
            k1 = self._f(X, u, i)
            k2 = self._f(X + 0.5 * self.dt * k1, u, i)
            k3 = self._f(X + 0.5 * self.dt * k2, u, i)
            k4 = self._f(X + self.dt * k3, u, i)
            X = X + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            X[4] = k1[3]
            traj[:, i + 1] = X.astype(np.float32)

        self.state = {"traj": traj, "t": 0.0}
        self.control = np.zeros(self.n_total)
        return self._get_obs(), {}