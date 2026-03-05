# train_sb3_ppo.py
# PPO Stable-Baselines3 sur ton SingleLaneEnv (obs Dict: traj, t)
#
# Prérequis:
#   pip install stable-baselines3[extra] gymnasium tensorboard
#
# Run:
#   python train_sb3_ppo.py
#   tensorboard --logdir ./runs

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from envs import SingleLaneEnv, BasisActionWrapper  # <-- /mnt/data/envs.py


def make_env(seed: int = 0, K: int = 10, T: float = 40.0, dt: float = 0.05, macro_dt: float = 1.0, al = 8):
    def _thunk():
        env = SingleLaneEnv(T=T, dt=dt, macro_dt=macro_dt)
        env = BasisActionWrapper(env, K=K) 
        env.reset(seed=seed)
        return env
    return _thunk


if __name__ == "__main__":
    exp_n = 3
    run_dir = os.path.expanduser(f"~/tb_logs/runs/ppo_singlelane_stable_{exp_n}")
    os.makedirs(run_dir, exist_ok=True)

    # ====== ENV TRAIN (vectorisé) ======
    n_envs = 1 
    train_env = DummyVecEnv([make_env(seed=42 + i) for i in range(n_envs)])
    train_env = VecMonitor(train_env, filename=os.path.join(run_dir, "monitor.csv"))

    # Normalisation obs/reward
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ====== ENV EVAL ======
    eval_env = DummyVecEnv([make_env(seed=123)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # synchro stats de normalisation eval <- train
    eval_env.obs_rms = train_env.obs_rms

    # ====== LOGGER ======
    logger = configure(run_dir, ["stdout", "tensorboard"])

    # ====== PPO (MultiInputPolicy pour obs Dict) ======
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.7,
        max_grad_norm=0.5,
        tensorboard_log=run_dir,
        verbose=1,
        device="auto",
        # policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))  # optionnel
    )
    model.set_logger(logger)

    # ====== CALLBACKS ======
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=run_dir,
        name_prefix="ppo_ckpt",
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    # ====== TRAIN ======
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb], progress_bar=True)

    # ====== SAVE FINAL ======
    model.save(os.path.join(run_dir, "ppo_final"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    print("\n✅ Training done.")
    print(f"Logs: {run_dir}")
    print(f"Tensorboard: tensorboard --logdir {run_dir}")

    # ====== QUICK ROLLOUT (1 épisode) ======
    # Pour évaluer correctement, recharge VecNormalize en mode eval (reward non normalisé)
    env = DummyVecEnv([make_env(seed=999)])
    env = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    done = False
    ep_rew = 0.0
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # VecEnv -> done bool (term/trunc fusionnés)
        ep_rew += float(reward)
        if done:
            break
    print(f"Eval episode reward: {ep_rew:.3f}")