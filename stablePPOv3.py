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
import time
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn

from envsv3 import SingleLaneEnv, BasisActionWrapper, SafetyWrapper  # <-- /mnt/data/envs.py


def make_env(seed: int = 0, K: int = 20, T: float = 40.0, dt: float = 0.05, macro_dt: float = 1.0, al: int = 8,
             alphaV: float = 2.0, use_safety: bool = False, h0: float = 2.0, alphaV_safe: float = 1.0, beta: float = 0.5):
    """
    use_safety=False  ->  identical to exp_n <= 6, SafetyWrapper not applied
    use_safety=True   ->  stacks SafetyWrapper with CBF gain beta
    """
    def _thunk():
        env = SingleLaneEnv(T=T, dt=dt, macro_dt=macro_dt, al=al, alphaV=alphaV)
        env = BasisActionWrapper(env, K=K)
        if use_safety:
            env = SafetyWrapper(env, h0=h0, alphaV=alphaV_safe, beta=beta)
        env.reset(seed=seed)
        return env
    return _thunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on SingleLaneEnv")
    # experiment
    parser.add_argument("--exp_n",        type=int,   default=3)
    parser.add_argument("--total_steps",  type=int,   default=750_000)
    parser.add_argument("--n_envs",       type=int,   default=1)
    # env
    parser.add_argument("--T",            type=float, default=40.0)
    parser.add_argument("--dt",           type=float, default=0.05)
    parser.add_argument("--macro_dt",     type=float, default=1.0)
    parser.add_argument("--K",            type=int,   default=20)
    parser.add_argument("--al",           type=int,   default=8)
    parser.add_argument("--alphaV",       type=float, default=2.0,  help="OV model gain in SingleLaneEnv")
    # safety wrapper
    parser.add_argument("--use_safety",   action="store_true", default=True)
    parser.add_argument("--no_safety",    action="store_true", default=False)
    parser.add_argument("--h0",           type=float, default=2.0)
    parser.add_argument("--alphaV_safe",  type=float, default=1.0,  help="CBF gain in SafetyWrapper")
    parser.add_argument("--beta",         type=float, default=1.0,  help="CBF derivative gain")
    # PPO
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--n_steps",      type=int,   default=4096)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--n_epochs",     type=int,   default=10)
    parser.add_argument("--ent_coef",     type=float, default=0.01)
    args = parser.parse_args()

    use_safety = args.use_safety and not args.no_safety

    exp_n = args.exp_n
    run_dir = os.path.expanduser(f"~/tb_logs/runs/ppo_singlelane_stable_saf_{exp_n}")
    os.makedirs(run_dir, exist_ok=True)

    env_kwargs = dict(K=args.K, T=args.T, dt=args.dt, macro_dt=args.macro_dt, al=args.al,
                      alphaV=args.alphaV, use_safety=use_safety, h0=args.h0,
                      alphaV_safe=args.alphaV_safe, beta=args.beta)

    # ====== ENV TRAIN (vectorisé) ======
    n_envs = args.n_envs
    train_env = DummyVecEnv([make_env(seed=42 + i, **env_kwargs) for i in range(n_envs)])
    train_env = VecMonitor(train_env, filename=os.path.join(run_dir, "monitor.csv"))

    # Normalisation obs/reward
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ====== ENV EVAL ======
    eval_env = DummyVecEnv([make_env(seed=123, **env_kwargs)])
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
        learning_rate=get_linear_fn(args.lr, 1e-5, 1.0),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.7,
        max_grad_norm=0.5,
        tensorboard_log=run_dir,
        verbose=1,
        device="auto",
        policy_kwargs=dict(
        net_arch=dict(pi=[512,512, 256], vf=[512,512, 256])
    ),
    )
    model.set_logger(logger)

    # ====== CALLBACKS ======
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000//n_envs,
        save_path=run_dir,
        name_prefix="ppo_ckpt",
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_safe"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=25_000//n_envs,  # éval tous les 25k steps (tous les 25k//n_envs updates)
        deterministic=True,
        render=False,
    )

    # ====== TRAIN ======
    total_timesteps = args.total_steps
    print(f"\n[train] Starting — {total_timesteps:,} timesteps, n_envs={n_envs}")
    print(f"[train] n_steps={model.n_steps}, batch_size={model.batch_size}, "
          f"n_epochs={model.n_epochs}")
    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb], progress_bar=True)
    elapsed = time.time() - t_start
    print(f"\n[train] Done in {elapsed/60:.1f} min  ({total_timesteps/elapsed:.0f} steps/sec)")

    # ====== SAVE FINAL ======
    model.save(os.path.join(run_dir, "ppo_safe_final"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    print("\n✅ Training done.")
    print(f"Logs: {run_dir}")
    print(f"Tensorboard: tensorboard --logdir {run_dir}")

    # ====== QUICK ROLLOUT (1 épisode) ======
    # Pour évaluer correctement, recharge VecNormalize en mode eval (reward non normalisé)
    env = DummyVecEnv([make_env(seed=999, K=20, use_safety=True, beta=1.0)])
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