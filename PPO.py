import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from collections import deque
from tqdm.auto import trange
import time
from torch.utils.tensorboard import SummaryWriter
from envs import SingleLaneEnv, BasisActionWrapper

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(64, act_dim)
        self.logstd_head = nn.Linear(64, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_head(x)
        logstd = self.logstd_head(x).clamp(-20, 2)
        std = logstd.exp()
        return mu, std
    
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
    


def PPO(
    env,
    episodes=100,
    n_epochs=4,
    batch_size=32,
    lr=3e-4,
    gamma=0.99,
    clip_eps=0.2,
    log_dir="runs/ppo_singlelane",
    log_every=1,
):
    writer = SummaryWriter(log_dir=log_dir)

    traj_dim = int(np.prod(env.observation_space["traj"].shape))  # 5*n_total
    obs_dim = traj_dim + 1  # + t
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    global_step = 0

    pbar = trange(episodes)
    for ep in pbar:
        obs, _ = env.reset()
        
        obs_buf, act_buf, rew_buf, done_buf, logp_old_buf, val_buf = [], [], [], [], [], []

        ep_return = 0.0
        base_env = env.unwrapped

        for _ in range(base_env.n_chunks):
            traj_flat = obs["traj"].reshape(-1).astype(np.float32)
            t = np.array([obs["t"]], dtype=np.float32)  # if obs["t"] is scalar
            obs_vec = np.concatenate([traj_flat, t], axis=0)

            obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)

            with torch.no_grad():
                mu, std = actor(obs_tensor)
                dist = Normal(mu, std)
                action = dist.sample()  # (1, act_dim)
                logp = dist.log_prob(action).sum(dim=-1)  # (1,)
                value = critic(obs_tensor)  # (1,)

            # step env
            step_out = env.step(action.squeeze(0).cpu().numpy())
            # your env returns: ((obs, info), reward, done, info)
            obs_out, reward, done, info = step_out
            next_obs = obs_out[0]  # Extract the observation from the tuple returned by env.step()

            # store
            obs_buf.append(obs_vec)
            act_buf.append(action.squeeze(0).cpu().numpy())
            rew_buf.append(float(reward))
            done_buf.append(float(done))
            logp_old_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))

            ep_return += float(reward)
            global_step += 1

            obs = next_obs
            if done:
                break

        # ---- tensors ----
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)          # (T, obs_dim)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)          # (T, act_dim)
        rew_t = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device)          # (T,)
        done_t = torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)        # (T,)
        logp_old_t = torch.tensor(np.array(logp_old_buf), dtype=torch.float32, device=device)# (T,)
        val_t = torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)

        # ---- returns (simple MC) ----
        returns = torch.zeros_like(rew_t)
        G = 0.0
        for i in reversed(range(len(rew_t))):
            G = rew_t[i] + gamma * G * (1.0 - done_t[i])
            returns[i] = G

        # ---- advantages ----
        adv = returns - val_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize helps PPO a lot

        # ---- PPO updates ----
        # we’ll log averages over all minibatches
        pi_loss_acc = 0.0
        v_loss_acc = 0.0
        kl_acc = 0.0
        clipfrac_acc = 0.0
        ent_acc = 0.0
        std_acc = 0.0
        n_mb = 0

        # indices for minibatches
        T = obs_t.shape[0]
        idxs = np.arange(T)

        for _ in range(n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, batch_size):
                mb = idxs[start:start + batch_size]
                batch_obs = obs_t[mb]         # (B, obs_dim)
                batch_act = act_t[mb]         # (B, act_dim)
                batch_logp_old = logp_old_t[mb] # (B,)
                batch_adv = adv[mb]           # (B,)
                batch_ret = returns[mb]       # (B,)

                mu, std = actor(batch_obs)
                dist = Normal(mu, std)
                logp_new = dist.log_prob(batch_act).sum(dim=-1)  # (B,)

                ratio = torch.exp(logp_new - batch_logp_old)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred = critic(batch_obs)
                critic_loss = nn.MSELoss()(value_pred, batch_ret)

                loss = actor_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
                optimizer.step()

                # ---- diagnostics (no grad) ----
                with torch.no_grad():
                    approx_kl = (batch_logp_old - logp_new).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
                    entropy = dist.entropy().sum(dim=-1).mean()
                    mean_std = std.mean()

                pi_loss_acc += float(actor_loss.item())
                v_loss_acc += float(critic_loss.item())
                kl_acc += float(approx_kl.item())
                clipfrac_acc += float(clipfrac.item())
                ent_acc += float(entropy.item())
                std_acc += float(mean_std.item())
                n_mb += 1

        # ---- TB logging (per episode) ----
        if (ep % log_every) == 0:
            writer.add_scalar("rollout/ep_return", ep_return, ep)
            writer.add_scalar("rollout/ep_len", T, ep)

            writer.add_scalar("loss/policy", pi_loss_acc / max(n_mb, 1), ep)
            writer.add_scalar("loss/value", v_loss_acc / max(n_mb, 1), ep)

            writer.add_scalar("diagnostics/approx_kl", kl_acc / max(n_mb, 1), ep)
            writer.add_scalar("diagnostics/clipfrac", clipfrac_acc / max(n_mb, 1), ep)
            writer.add_scalar("diagnostics/entropy", ent_acc / max(n_mb, 1), ep)
            writer.add_scalar("diagnostics/mean_std", std_acc / max(n_mb, 1), ep)

        pbar.set_postfix({
            "R": f"{ep_return:.2f}",
            "KL": f"{(kl_acc/max(n_mb,1)):.4f}",
            "clip%": f"{100*(clipfrac_acc/max(n_mb,1)):.1f}",
        })

    writer.close()
    return actor, critic

if __name__ == "__main__":
    base_env = SingleLaneEnv(T=50.0, dt = 0.01, macro_dt=0.5)

    K_value = base_env.n_chunks

    wrapped_env = BasisActionWrapper(base_env, K=K_value)

    wrapped_env.n_chunks = K_value

    actor, critic = PPO(wrapped_env, episodes=100, log_dir="runs/ppo_singlelane")