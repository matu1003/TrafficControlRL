import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class SingleLaneEnv(gym.Env):
    """
    A simple single-lane driving environment.
    The agent can accelerate, decelerate, or maintain speed to reach a target position.
    """

    def __init__(self,
                 xl0 = 1.0,
                 vl0 = 15.0,
                 x0 = 0.0,
                 v0 = 10.0,
                 dt = 0.05,
                 macro_dt = 0.5,
                 T = 20.0,
                 v_min = 0.0,
                 v_max = 30.0,
                 a_min=-3.0,
                 a_max=2.0,
                 u_min=-3.0,
                 u_max=2.0,
                 alpha_a=1.0,
                 alpha_v=0.2,
                 alpha_d=2.0,
                 alphaV=0.4
                 ):
        super(SingleLaneEnv, self).__init__()
        self.xl0 = float(xl0)
        self.vl0 = float(vl0)
        self.x0 = float(x0)
        self.v0 = float(v0)
        self.dt = float(dt)
        self.macro_dt = float(macro_dt)
        self.chunk_micro = int(np.round(self.macro_dt / self.dt))
        self.T = float(T)
        self.alphaV = float(alphaV)
        self.n_internal = int(np.round(self.macro_dt / self.dt))
        self.n_total = int(np.round(self.T / self.dt))
        
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.a_min, self.a_max = float(a_min), float(a_max)
        self.u_min, self.u_max = float(u_min), float(u_max)

        self.action_space = gym.spaces.Box(
        low = np.full((self.n_total,), self.u_min, dtype=np.float32),
        high = np.full((self.n_total,), self.u_max, dtype=np.float32),
        dtype = np.float32
        )

        obs_low = np.array([
            np.full((self.n_total,), -1e6, dtype=np.float32),
            np.full((self.n_total,), self.v_min, dtype=np.float32),
            np.full((self.n_total,), -1e6, dtype=np.float32),
            np.full((self.n_total,), self.v_min, dtype=np.float32),
            np.full((self.n_total,), self.a_min, dtype=np.float32)], dtype=np.float32)
        obs_high = np.array([
            np.full((self.n_total,), 1e6, dtype=np.float32),
            np.full((self.n_total,), self.v_max, dtype=np.float32),
            np.full((self.n_total,), 1e6, dtype=np.float32),
            np.full((self.n_total,), self.v_max, dtype=np.float32),
            np.full((self.n_total,), self.a_max, dtype=np.float32)], dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "traj": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32, shape=(5, self.n_total)),
            "t": gym.spaces.Box(low=0.0, high=self.T, shape=(1,), dtype=np.float32),
        })

        traj = np.zeros((5, self.n_internal * self.n_total), dtype=np.float64)
        traj[:, 0] = np.array([xl0, vl0, x0, v0, 0], dtype=np.float64)
        print(traj.shape)
        self.state = {"traj": traj, "t": 0.0}

        self.alpha_a = float(alpha_a)
        self.alpha_v = float(alpha_v)
        self.alpha_d = float(alpha_d)
        self.V = lambda x: np.tanh(x-2) + np.tanh(2)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        traj = np.zeros((5, self.n_total), dtype=np.float64)
        traj[:, 0] = np.array([self.xl0, self.vl0, self.x0, self.v0, 0], dtype=np.float64)

        self.state = {"traj": traj, "t": 0.0}

        return self._get_obs()

    def step(self, action):
        '''
        Aplly action on a macro time step through rk4 integration starting from curent state and time
        :param action: Control action to be applied.
        '''
        a_profile = np.clip(action, self.u_min, self.u_max)
        X0 = self.state['traj'][:, int(self.state['t'] / self.dt)]
        t0 = self.state['t']
        k0 = int(round(self.state["t"] / self.dt))
        k1 = min(k0 + self.n_internal, self.n_total-1)
        remaining_steps = k1 - k0 

        newtraj = np.zeros((5, self.n_total), dtype=np.float64)
        newtraj[:, :k0+1] = self.state['traj'][:, :k0+1]

        X = X0.copy()
        for i in range(remaining_steps):
            u = a_profile[k0 + i]
            k1 = self._f(X, u)
            k2 = self._f(X + 0.5 * self.dt * k1, u)
            k3 = self._f(X + 0.5 * self.dt * k2, u)
            k4 = self._f(X + self.dt * k3, u)
            X += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            newtraj[:, k0+i+1] = X

        self.state["traj"] = newtraj
        self.state["t"] = (k0 + remaining_steps) * self.dt

        if self.state["t"] >= self.T:
            done = True
        else:
            done = False
        
        reward = self.compute_reward()

        return self._get_obs(), reward, done, {}

    def _f(self, s: np.ndarray, u: float):
        '''
        Transition function for the system dynamics.
        
        :param s: state vector (x_lead, v_lead, x, v, a)
        :param u: control input (1,)
        '''
        xL, vL, x, v, a = s
        dxL = vL
        dvL = 0.0
        dx = v
        dv = self.alphaV * ( self.V(xL - x) - v) + u
        da = 0.0
        return np.array([dxL, dvL, dx, dv, da], dtype=np.float64)
    
    def compute_reward(self):
        xL, vL, x, v, a = self.state["traj"]
        
        integrand = 0.5 * self.alpha_a * (a**2) - self.alpha_v * v
        
        reward = -np.trapezoid(integrand, dx=self.dt) / self.T - self.alpha_d * (vL[0] - vL[-1])**2
        
        return reward

    def _get_obs(self):
        return self.state, {}

    def render(self, control):
        N = self.n_total

        t = np.arange(N) * self.dt
        k = int(np.round(self.state["t"] / self.dt))


        xL, vL, x, v, a = self.state["traj"]


        # create figure once
        if not hasattr(self, "_fig"):
            self._fig, self._axs = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True
            )
        plt.ion()


        ax_pos, ax_vel, ax_u = self._axs
        ax_pos.clear()
        ax_vel.clear()
        ax_u.clear()


        # ---------- positions ----------
        ax_pos.plot(t[:k+1], xL[:k+1], "C0-", label="Leader (past)")
        ax_pos.plot(t[:k+1], x[:k+1], "C1-", label="Ego (past)")


        ax_pos.plot(t[k:], xL[k:], "C0--", alpha=0.6, label="Leader (projected)")
        ax_pos.plot(t[k:], x[k:], "C1--", alpha=0.6, label="Ego (projected)")


        ax_pos.axvline(t[k], color="k", linestyle=":", label="current time")
        ax_pos.set_ylabel("Position")
        ax_pos.legend(loc="upper left")
        ax_pos.grid(True)


        # ---------- velocities ----------
        ax_vel.plot(t[:k+1], vL[:k+1], "C0-")
        ax_vel.plot(t[:k+1], v[:k+1], "C1-")


        ax_vel.plot(t[k:], vL[k:], "C0--", alpha=0.6)
        ax_vel.plot(t[k:], v[k:], "C1--", alpha=0.6)


        ax_vel.axvline(t[k], color="k", linestyle=":")
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True)


        # ---------- control ----------
        ax_u.plot(t[:k+1], control[:k+1], "C2-", label="planned control")


        # highlight last applied chunk
        k0 = max(0, k - self.chunk_micro)
        ax_u.axvspan(
        t[k0], t[k],
        color="C2", alpha=0.2, label="applied chunk"
        )


        ax_u.axvline(t[k], color="k", linestyle=":")
        ax_u.set_ylabel("Acceleration")
        ax_u.set_xlabel("Time [s]")
        ax_u.grid(True)
        ax_u.legend(loc="upper right")


        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        clear_output(wait=True)
        display(self._fig)

    def close(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt