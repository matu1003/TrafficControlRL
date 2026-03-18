"""
train_chain_ppo.py
==================
Sequential multi-vehicle chain training with PPO (Stable-Baselines3).

Chain structure
---------------
  Vehicle 0   ←  random degree-≤3 polynomial leader (no model)
  Vehicle 1   ←  follower of vehicle 0   [PPO pair_0]
  Vehicle 2   ←  follower of vehicle 1   [PPO pair_1, frozen pair_0]
  Vehicle 3   ←  follower of vehicle 2   [PPO pair_2, frozen pair_1]

Training is strictly sequential:
  1. Pre-generate 500 polynomial profiles  → datasets/pair0_leaders.npy
  2. Train pair_0 (vehicle 1 samples from that dataset at O(1) reset cost).
  3. Freeze pair_0 best checkpoint.
  4. Pre-generate 500 rollout profiles of pair_0  → datasets/pair1_leaders.npy
  5. Train pair_1.  Repeat for pair_2.

Reset cost at training time
---------------------------
  Old approach  : O(n_total) RK4 steps per rollout × chain depth
  New approach  : O(1) numpy array index + O(n_total) RK4 for the
                  follower's own reference trajectory — same as base env.

Usage
-----
  python train_chain_ppo.py

Tensorboard
-----------
  tensorboard --logdir ~/tb_logs/chain
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn

from envs import BasisActionWrapper
from envs_chain import (
    DatasetLeaderEnv,
    generate_leader_dataset,
)


# ============================================================
# Global hyperparameters
# ============================================================

BASE_DIR     = os.path.expanduser("~/tb_logs/runs/chain2")
DATASET_DIR  = os.path.join(BASE_DIR, "datasets")
N_PAIRS      = 3           # number of follower vehicles
N_DATASET    = 500         # leader profiles pre-generated per pair
K            = 20          # basis action coefficients
POLY_DEGREE  = 3

ENV_KWARGS = dict(
    xl0=1.0, vl0=15.0, x0=0.0, v0=10.0,
    dt=0.05, macro_dt=1.0, T=40.0,
    v_min=0.0, v_max=30.0,
    a_min=-9.0, a_max=4.0,
    u_min=-9.0, u_max=4.0,
    alpha_a=1.0, alpha_v=1.0, alpha_d=1.0,
    alphaV=0.4,
)

PPO_KWARGS = dict(
    policy="MultiInputPolicy",
    learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
    n_steps=4096,
    batch_size=512,
    n_epochs=5,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.7,
    max_grad_norm=0.5,
    verbose=1,
    device="auto",
    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
)

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ       = 25_000
CKPT_FREQ       = 50_000


# ============================================================
# Environment factories
# ============================================================

def make_dataset_env(dataset_path: str, seed: int):
    """Factory: follower env whose leader is drawn from a pre-built dataset."""
    def _thunk():
        env = DatasetLeaderEnv(dataset_path=dataset_path, **ENV_KWARGS)
        env = BasisActionWrapper(env, K=K)
        env.reset(seed=seed)
        return env
    return _thunk


# ============================================================
# Training helper
# ============================================================

def train_pair(pair_idx: int,
               dataset_path: str,
               run_dir: str):
    """
    Train one PPO follower model for TOTAL_TIMESTEPS steps.
    Leader trajectories are drawn O(1) from `dataset_path`.

    Returns
    -------
    model    : PPO
    train_vn : VecNormalize  (passed to generate_leader_dataset next stage)
    """
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Training PAIR {pair_idx}  →  {run_dir}")
    print(f"  Leader dataset : {dataset_path}  ({N_DATASET} profiles)")
    print(f"{'='*60}\n")

    # ---- train env ----
    train_env = DummyVecEnv([make_dataset_env(dataset_path, seed=42 + i)
                             for i in range(1)])
    train_env = VecMonitor(train_env,
                           filename=os.path.join(run_dir, "monitor.csv"))
    train_env = VecNormalize(train_env,
                             norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---- eval env ----
    eval_env = DummyVecEnv([make_dataset_env(dataset_path, seed=123)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env,
                            norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.obs_rms = train_env.obs_rms

    # ---- logger ----
    logger = configure(run_dir, ["stdout", "tensorboard"])

    # ---- model ----
    model = PPO(env=train_env, tensorboard_log=run_dir, **PPO_KWARGS)
    model.set_logger(logger)

    # ---- callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=run_dir,
        name_prefix=f"pair{pair_idx}_ckpt",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # ---- train ----
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ---- save ----
    model.save(os.path.join(run_dir, f"pair{pair_idx}_final"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    print(f"\n✅  Pair {pair_idx} done.  Best model → {run_dir}/best/")
    return model, train_env


def load_frozen(run_dir: str, pair_idx: int, dataset_path: str):
    """
    Load the best checkpoint and its VecNormalize stats.
    Returns (model, vec_normalize) ready for generate_leader_dataset().
    """
    best_model_path = os.path.join(run_dir, "best", "best_model")
    vn_path         = os.path.join(run_dir, "vecnormalize.pkl")

    print(f"\n🔒  Freezing pair {pair_idx}: {best_model_path}")

    dummy_env = DummyVecEnv([make_dataset_env(dataset_path, seed=0)])
    vn = VecNormalize.load(vn_path, dummy_env)
    vn.training    = False
    vn.norm_reward = False

    model = PPO.load(best_model_path, env=vn)
    return model, vn


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(BASE_DIR,    exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # PAIR 0  –  vehicle 1 follows polynomial vehicle 0
    # ----------------------------------------------------------------

    # Step 1: pre-generate polynomial leader profiles (no model needed)
    ds0 = os.path.join(DATASET_DIR, "pair0_leaders.npy")
    generate_leader_dataset(
        save_path=ds0,
        n=N_DATASET,
        env_kwargs=ENV_KWARGS,
        model=None,           # polynomial mode
        poly_degree=POLY_DEGREE,
        base_seed=0,
    )

    # Step 2: train pair 0
    pair0_dir = os.path.join(BASE_DIR, "pair_0")
    model0, vn0 = train_pair(pair_idx=0,
                              dataset_path=ds0,
                              run_dir=pair0_dir)

    # Step 3: freeze pair 0
    frozen_model0, frozen_vn0 = load_frozen(pair0_dir, pair_idx=0,
                                             dataset_path=ds0)

    # ----------------------------------------------------------------
    # PAIR 1  –  vehicle 2 follows frozen vehicle 1
    # ----------------------------------------------------------------

    # Step 4: roll out frozen pair 0 × 500 → pair 1's leader dataset
    ds1 = os.path.join(DATASET_DIR, "pair1_leaders.npy")
    generate_leader_dataset(
        save_path=ds1,
        n=N_DATASET,
        env_kwargs=ENV_KWARGS,
        model=frozen_model0,
        vec_norm=frozen_vn0,
        K=K,
        base_seed=1000,
    )

    # Step 5: train pair 1
    pair1_dir = os.path.join(BASE_DIR, "pair_1")
    model1, vn1 = train_pair(pair_idx=1,
                              dataset_path=ds1,
                              run_dir=pair1_dir)

    # Step 6: freeze pair 1
    frozen_model1, frozen_vn1 = load_frozen(pair1_dir, pair_idx=1,
                                             dataset_path=ds1)

    # ----------------------------------------------------------------
    # PAIR 2  –  vehicle 3 follows frozen vehicle 2
    # ----------------------------------------------------------------

    # Step 7: roll out frozen pair 1 × 500 → pair 2's leader dataset
    ds2 = os.path.join(DATASET_DIR, "pair2_leaders.npy")
    generate_leader_dataset(
        save_path=ds2,
        n=N_DATASET,
        env_kwargs=ENV_KWARGS,
        model=frozen_model1,
        vec_norm=frozen_vn1,
        K=K,
        base_seed=2000,
    )

    # Step 8: train pair 2
    pair2_dir = os.path.join(BASE_DIR, "pair_2")
    model2, vn2 = train_pair(pair_idx=2,
                              dataset_path=ds2,
                              run_dir=pair2_dir)

    # ----------------------------------------------------------------
    # FINAL EVALUATION  –  one deterministic episode per vehicle
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  Full chain evaluation")
    print("="*60)

    frozen_model2, frozen_vn2 = load_frozen(pair2_dir, pair_idx=2,
                                             dataset_path=ds2)

    for pid, (m, ds) in enumerate([
            (frozen_model0, ds0),
            (frozen_model1, ds1),
            (frozen_model2, ds2),
    ]):
        ev = DummyVecEnv([make_dataset_env(ds, seed=999)])
        vn_path = os.path.join(BASE_DIR, f"pair_{pid}", "vecnormalize.pkl")
        ev = VecNormalize.load(vn_path, ev)
        ev.training    = False
        ev.norm_reward = False

        obs = ev.reset()
        done = False
        ep_rew = 0.0
        for _ in range(10_000):
            action, _ = m.predict(obs, deterministic=True)
            obs, reward, dones, _ = ev.step(action)
            ep_rew += float(reward)
            if dones[0]:
                break
        print(f"  Vehicle {pid+1}  episode reward: {ep_rew:.3f}")

    print(f"\n✅  Chain training complete.")
    print(f"  Logs     : {BASE_DIR}")
    print(f"  Datasets : {DATASET_DIR}")
    print(f"  Tensorboard: tensorboard --logdir {BASE_DIR}")