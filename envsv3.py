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
                 dt = 0.01,
                 macro_dt = 0.5,
                 T = 20.0,
                 v_min = 0.0,
                 v_max = 30.0,
                 a_min=-9.0,
                 a_max=4.0,
                 u_min=-9.0,
                 u_max=4.0,
                 alpha_a=1.0,
                 alpha_v=1.0,
                 alpha_d=1.0,
                 alphaV=2,
                 al = 3
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
        self.n_chunks = self.n_total // self.n_internal
        
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
            np.full((self.n_total,), self.a_max, dtype=np.float32)])
        self.observation_space = gym.spaces.Dict({
            "traj": gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32, shape=(5, self.n_total)),
            "t": gym.spaces.Box(low=0.0, high=self.T, shape=(1,), dtype=np.float32),
        })

        traj = np.zeros((5, self.n_internal * self.n_total), dtype=np.float64)
        traj[:, 0] = np.array([xl0, vl0, x0, v0, 0], dtype=np.float64)
        self.state = {"traj": traj, "t": 0.0}

        t = np.arange(self.n_total)


        
        self.leading_al =  a_max/2 * (np.sin(al*t*np.pi/(self.n_total)))

        self.control = np.zeros(self.n_total)
        
        self.alpha_a = float(alpha_a)
        self.alpha_v = float(alpha_v)
        self.alpha_d = float(alpha_d)

        t0 = np.tanh(1.0)
        g0 = (self.v_max - self.v_min) # Caracteristic gap = 2 seconds safety distance at avg speed
        self.V = lambda x: self.v_min + (self.v_max - self.v_min) * (np.tanh((x-g0)/g0) + t0) / (1 + t0)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        traj = np.zeros((5, self.n_total), dtype=np.float32)
        traj[:, 0] = np.array([self.xl0, self.vl0, self.x0, self.v0, 0], dtype=np.float32)

        X = traj[:, 0].astype(np.float64, copy=True)

        for i in range(self.n_total - 1):
            u = 0.0
            k1 = self._f(X, u, i)
            k2 = self._f(X + 0.5 * self.dt * k1, u, i)
            k3 = self._f(X + 0.5 * self.dt * k2, u, i)
            k4 = self._f(X + self.dt * k3, u, i)
            X = X + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            X[4] = k1[3]

            traj[:, i + 1] = X.astype(np.float32, copy=False)

        self.state = {"traj": traj, "t": 0.0}
        info = {}
        return self._get_obs(), info

    def step(self, action):
        '''
        Aplly action on a macro time step through rk4 integration starting from curent state and time
        :param action: Control action to be applied.
        '''
        a_profile = np.clip(action, self.u_min, self.u_max)
        X0 = self.state['traj'][:, int(self.state['t'] / self.dt)]
        t0 = self.state['t']
        p0 = int(round(t0 / self.dt))
        p1 = int(round((t0 + self.macro_dt) / self.dt))
        pf = self.n_total - 1
        remaining_steps = pf - p0

        newtraj = np.zeros((5, self.n_total), dtype=np.float64)
        newtraj[:, :p0+1] = self.state['traj'][:, :p0+1]
        self.control[p0+1:] = action[p0+1:]

        reward = 0.0
        done = newtraj[0, p0] < newtraj[2, p0]
        if done:
            return self._get_obs(), -1e2, True, False, {}

        X = X0.copy()
        for i in range(remaining_steps):
            u = a_profile[p0 + i]
            k1 = self._f(X, u, p0+i)
            k2 = self._f(X + 0.5 * self.dt * k1, u, p0+i)
            k3 = self._f(X + 0.5 * self.dt * k2, u, p0+i)
            k4 = self._f(X + self.dt * k3, u, p0+i)
            X += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            X[4] = k1[3]
            newtraj[:, p0+i+1] = X


        self.state["traj"] = newtraj
        self.state["t"] = p1 * self.dt

        if self.state["t"] >= self.T:
            done = True
        
        reward += self.compute_reward()

        terminated = done

        truncated = self.state["t"] >= self.T

        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def _f(self, s: np.ndarray, u: float, i: int = 0):
        '''
        Transition function for the system dynamics.
        
        :param s: state vector (x_lead, v_lead, x, v, a)
        :param u: control input (1,)
        '''
        xL, vL, x, v, a = s
        dxL = vL
        dvL = self.leading_al[i]
        dx = v
        dv = self.alphaV * ( self.V(xL - x) - v) + u + 
        # dv = u
        da = 0.0
        return np.array([dxL, dvL, dx, dv, da], dtype=np.float64)
    
    def compute_reward(self):
        xL, vL, x, v, a = self.state["traj"]
        
        integrand = 0.5 * self.alpha_a * ((a/self.a_max)**2) - self.alpha_v * v/self.v_max + self.alpha_d * ((vL - v)/self.v_max)**2
        
        reward = -np.trapz(integrand, dx=self.dt) / self.T 
        return reward


    def _get_obs(self):
        traj = self.state["traj"].astype(np.float32, copy=False)
        t = np.array([self.state["t"]], dtype=np.float32)
        return {"traj": traj, "t": t}

    def render(self):
        N = self.n_total

        t = np.arange(N) * self.dt
        k = int(round(self.state["t"] / self.dt))
        if k >= N:
            return


        xL, vL, x, v, a = self.state["traj"]


        # create figure once
        if not hasattr(self, "_fig"):
            self._fig = plt.figure(figsize=(10,8))
            gs = self._fig.add_gridspec(4,1)
            ax_pos = self._fig.add_subplot(gs[0])
            ax_vel = self._fig.add_subplot(gs[1], sharex=ax_pos)
            ax_u   = self._fig.add_subplot(gs[2], sharex=ax_pos)
            ax_phase = self._fig.add_subplot(gs[3]) 
            self._axs = (ax_pos, ax_vel, ax_u, ax_phase)
        plt.ion()


        ax_pos, ax_vel, ax_u, ax_phase = self._axs
        ax_pos.clear()
        ax_vel.clear()
        ax_u.clear()
        ax_phase.clear()


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
        uc = np.clip(self.control, self.u_min, self.u_max)
        ax_u.plot(t[:k+1], uc[:k+1], "C2-", label="Applied control")
        ax_u.plot(t[k:], uc[k:], "C2--", alpha=0.6, label="Planned control")

        # ---------- acceleration ----------
        ax_u.plot(t[:k+1], a[:k+1], "C3-", label="Accleration history")
        ax_u.plot(t[k:], a[k:], "C3--", alpha=0.6, label="Simulated acceleration")

        
        # ---------- leading acceleration ----------
        al = self.leading_al
        ax_u.plot(t[:k+1], al[:k+1], "C4-", label="Leading Accleration history")
        ax_u.plot(t[k:], al[k:], "C4--", alpha=0.6, label="Leading Simulated acceleration")

        # ----------- phase portrait (v, xL-x) ----------
        ax_phase.plot(xL[:k+1] - x[:k+1], vL[:k+1] - v[:k+1], "C1-", label="Phase portrait (past)")
        ax_phase.plot(xL[k:] - x[k:], vL[k:] - v[k:], "C1--", alpha=0.6, label="Phase portrait (projected)")
        ax_phase.axvline(0, color="k", linestyle=":")
        ax_phase.set_xlabel("Gap (xL - x)")
        ax_phase.set_ylabel("Velocity")
        ax_phase.legend(loc="upper right")
        ax_phase.grid(True)


        # highlight last applied chunk
        k0 = max(0, k - self.chunk_micro)
        ax_u.axvspan(
        t[k0], t[k],
        color="C2", alpha=0.2, label="applied chunk"
        )


        ax_u.axvline(t[k], color="k", linestyle=":")
        ax_u.set_ylabel("Acceleration/Control")
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


class BasisActionWrapper(gym.Wrapper):
    """
    PPO-compatible wrapper:
    - Action = alpha ∈ R^K (basis coefficients)
    - Internally builds u(t) over one macro step
    """

    def __init__(self, env: SingleLaneEnv, K: int):
        super().__init__(env)

        self.K = K
        self.tau = env.macro_dt
        self.dt = env.dt
        self.steps_per_macro = int(self.tau / self.dt)

        # PPO action space: coefficients alpha_i
        self.action_space = gym.spaces.Box(
            low=env.u_min,
            high=env.u_max,
            shape=(K,),
            dtype=np.float32
        )

        # observation unchanged
        self.observation_space = env.observation_space

        Phi = np.zeros((K, env.n_total), dtype=np.float32)

        # Bin edges in index space [0, n_total]
        edges = np.linspace(0, env.n_total, K + 1)
        edges = np.round(edges).astype(int)
        edges[0] = 0
        edges[-1] = env.n_total

        for i in range(K):
            a, b = edges[i], edges[i + 1]
            if b > a:
                Phi[i, a:b] = 1.0

        self.Phi = Phi

    def alpha_to_control(self, alpha, t0_index):
        """
        Build full control vector u(t) from alpha
        """
        u = alpha @ self.Phi
        return u

    def step(self, alpha):
        # current discrete time index
        t0_index = int(self.env.state["t"] / self.dt)

        # build control trajectory
        alpha = np.clip(alpha, self.env.u_min, self.env.u_max)
        control = self.alpha_to_control(alpha, t0_index)

        # delegate physics + reward to original env
        obs, reward, terminated, truncated, info = self.env.step(control)

        return obs, reward, terminated, truncated, info


class SafetyWrapper(gym.Wrapper):
    """
    Layered safety filter implementing:
        u(t) = min{ u_target(t), u_c(t), u_safe(t) }

    where:
        u_target = alphaV * (V(h) - v)          [OV model: soft following]
        u_safe   = alphaV * (v_safe - v) + dv_safe/dt  [CBF: hard safety]
        u_c      = RL policy output              [performance]

    v_safe(t) = sqrt( 2*|a_min| * (h - h0 + 0.5 * vL^2 / |aL_min|) )

    Parameters
    ----------
    h0        : minimum standstill gap (m), default 2.0
    alphaV    : gain for both u_target and u_safe, default 1.0
    """

    def __init__(self, env: gym.Env, h0: float = 2.0, alphaV: float = 1.0, beta: float = 1.0):
        super().__init__(env)
        self.h0 = float(h0)
        self.alphaV = float(alphaV)
        self.beta = float(beta)  # gain on the CBF derivative term d(v_safe)/dt

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _v_safe(self, h: float, vL: float, aL_min: float) -> float:
        """
        Maximum ego speed that guarantees no collision assuming worst-case
        leader braking at aL_min (negative value expected).
        """
        aL_min_abs = abs(aL_min)
        a_min_abs  = abs(self.unwrapped.a_min)  # ego max braking

        inner = h - self.h0 + 0.5 * vL**2 / (aL_min_abs + 1e-8)
        inner = max(inner, 0.0)           # clamp: never take sqrt of negative
        return np.sqrt(2.0 * a_min_abs * inner)

    def _u_safe(self, h: float, v: float, vL: float,
                aL_min: float, prev_vs: float, dt: float) -> float:
        """
        CBF-derived control:  u_safe = alphaV*(v_safe - v) + d(v_safe)/dt
        d(v_safe)/dt is approximated by finite difference with previous step.
        """
        vs = self._v_safe(h, vL, aL_min)
        dvs_dt = (vs - prev_vs) / dt
        return self.alphaV * (vs - v) + self.beta * dvs_dt

    def _u_target(self, h: float, v: float) -> float:
        """OV model: drive toward the desired speed V(gap)."""
        return self.alphaV * (self.unwrapped.V(h) - v)

    # ------------------------------------------------------------------
    # step: intercept action, apply filter, delegate to inner env
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped
        xL, vL, x, v, a = base.state["traj"][:, 0]
        aL_min = float(np.min(base.leading_al))
        self._prev_vs = self._v_safe(xL - x, vL, aL_min)
        self._safety_history = {"u_c": [], "u_safe": [], "u_target": [], "u_applied": [], "t": []}
        return obs, info

    def step(self, action):
        """
        action  : u_c control profile, shape (n_total,) or (K,) depending on
                  whether BasisActionWrapper is stacked below/above.
                  We filter pointwise at the current timestep only.
        """
        # current state
        base  = self.unwrapped
        p0    = int(round(base.state["t"] / base.dt))
        p0    = min(p0, base.n_total - 1)
        xL, vL, x, v, a = base.state["traj"][:, p0]
        h     = float(xL - x)
        v     = float(v)
        vL    = float(vL)

        # worst-case leader deceleration over remaining horizon
        p_end   = base.n_total
        aL_min  = float(np.min(base.leading_al[p0:p_end]))

        # compute the three candidate controls (scalar)
        u_target = self._u_target(h, v)
        u_safe   = self._u_safe(h, v, vL, aL_min, self._prev_vs, base.dt)
        if hasattr(action, '__len__') and len(action) < base.n_total:
            # action is K basis coefficients — map p0 to the corresponding bin
            K = len(action)
            bin_idx = min(int(p0 * K / base.n_total), K - 1)
            u_c = float(action[bin_idx])
        elif hasattr(action, '__len__'):
            u_c = float(action[p0])
        else:
            u_c = float(action)

        u_applied = min(u_target, u_c, u_safe)
        u_applied = float(np.clip(u_applied, base.u_min, base.u_max))

        # update v_safe memory for next d/dt estimate
        xL_n, vL_n, x_n, v_n, _ = base.state["traj"][:, min(p0+1, base.n_total-1)]
        self._prev_vs = self._v_safe(float(xL_n - x_n), float(vL_n), aL_min)

        # overwrite current-step control in the action vector
        if hasattr(action, '__len__') and len(action) < base.n_total:
            action_filtered = action.copy()
            action_filtered[bin_idx:] = u_applied  # clamp remaining bins
        elif hasattr(action, '__len__'):
            action_filtered = action.copy()
            action_filtered[p0:] = u_applied
        else:
            action_filtered = u_applied

        obs, reward, terminated, truncated, info = self.env.step(action_filtered)
        info["u_safe"]    = u_safe
        info["u_target"]  = u_target
        info["u_c"]       = u_c
        info["u_applied"] = u_applied

        if not hasattr(self, "_safety_history"):
            self._safety_history = {"u_c": [], "u_safe": [], "u_target": [], "u_applied": [], "t": []}
        self._safety_history["u_c"].append(u_c)
        self._safety_history["u_safe"].append(u_safe)
        self._safety_history["u_target"].append(u_target)
        self._safety_history["u_applied"].append(u_applied)
        self._safety_history["t"].append(p0 * base.dt)

        return obs, reward, terminated, truncated, info

    def render(self):
        base = self.unwrapped
        N  = base.n_total
        t  = np.arange(N) * base.dt
        k  = min(int(round(base.state["t"] / base.dt)), N - 1)

        xL, vL, x, v, a = base.state["traj"]
        al = base.leading_al

        if not hasattr(self, "_fig"):
            self._fig = plt.figure(figsize=(11, 12))
            gs = self._fig.add_gridspec(5, 1)
            self._axs = (
                self._fig.add_subplot(gs[0]),
                self._fig.add_subplot(gs[1]),
                self._fig.add_subplot(gs[2]),
                self._fig.add_subplot(gs[3]),
                self._fig.add_subplot(gs[4]),
            )
        plt.ion()

        ax_pos, ax_vel, ax_u, ax_phase, ax_safe = self._axs
        for ax in self._axs:
            ax.clear()

        # --- positions ---
        ax_pos.plot(t[:k+1], xL[:k+1], "C0-", label="Leader (past)")
        ax_pos.plot(t[:k+1],  x[:k+1], "C1-", label="Ego (past)")
        ax_pos.plot(t[k:], xL[k:], "C0--", alpha=0.6, label="Leader (projected)")
        ax_pos.plot(t[k:],  x[k:], "C1--", alpha=0.6, label="Ego (projected)")
        ax_pos.axvline(t[k], color="k", linestyle=":")
        ax_pos.set_ylabel("Position")
        ax_pos.legend(loc="upper left")
        ax_pos.grid(True)

        # --- velocities ---
        ax_vel.plot(t[:k+1], vL[:k+1], "C0-")
        ax_vel.plot(t[:k+1],  v[:k+1], "C1-")
        ax_vel.plot(t[k:], vL[k:], "C0--", alpha=0.6)
        ax_vel.plot(t[k:],  v[k:], "C1--", alpha=0.6)
        ax_vel.axvline(t[k], color="k", linestyle=":")
        ax_vel.set_ylabel("Velocity")
        ax_vel.grid(True)

        # --- control + acceleration ---
        uc = np.clip(base.control, base.u_min, base.u_max)
        ax_u.plot(t[:k+1], uc[:k+1], "C2-", label="Applied control")
        ax_u.plot(t[k:],   uc[k:],   "C2--", alpha=0.6, label="Planned control")
        ax_u.plot(t[:k+1],  a[:k+1], "C3-", label="Acceleration history")
        ax_u.plot(t[k:],    a[k:],   "C3--", alpha=0.6, label="Simulated acceleration")
        ax_u.plot(t[:k+1], al[:k+1], "C4-", label="Leading accel history")
        ax_u.plot(t[k:],   al[k:],   "C4--", alpha=0.6, label="Leading accel simulated")
        k0 = max(0, k - base.chunk_micro)
        ax_u.axvspan(t[k0], t[k], color="C2", alpha=0.2, label="applied chunk")
        ax_u.axvline(t[k], color="k", linestyle=":")
        ax_u.set_ylabel("Acceleration/Control")
        ax_u.legend(loc="upper right")
        ax_u.grid(True)

        # --- phase portrait ---
        ax_phase.plot(xL[:k+1] - x[:k+1], vL[:k+1] - v[:k+1], "C1-", label="Past")
        ax_phase.plot(xL[k:]   - x[k:],   vL[k:]   - v[k:],   "C1--", alpha=0.6, label="Projected")
        ax_phase.axvline(0, color="k", linestyle=":")
        ax_phase.set_xlabel("Gap (xL - x)")
        ax_phase.set_ylabel("Δv")
        ax_phase.legend(loc="upper right")
        ax_phase.grid(True)

        # --- safety filter history ---
        if hasattr(self, "_safety_history") and self._safety_history["t"]:
            sh = self._safety_history
            ax_safe.plot(sh["t"], sh["u_c"],       "C0-",  label="u_c (RL)")
            ax_safe.plot(sh["t"], sh["u_safe"],    "C3-",  label="u_safe (CBF)")
            ax_safe.plot(sh["t"], sh["u_target"],  "C4--", label="u_target (OV)")
            ax_safe.plot(sh["t"], sh["u_applied"], "C2-",  linewidth=2, label="u_applied")
        ax_safe.axvline(t[k], color="k", linestyle=":")
        ax_safe.set_ylabel("Control (m/s²)")
        ax_safe.set_xlabel("Time [s]")
        ax_safe.set_title("Safety filter")
        ax_safe.legend(loc="upper right")
        ax_safe.grid(True)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        clear_output(wait=True)
        display(self._fig)


if __name__ == "__main__":
    import matplotlib.pyplot as plt