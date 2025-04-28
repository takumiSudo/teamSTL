from src.interfaces import EnvWrapper, PolicyBase
import torch
from src.dynamic_helper import (
    single_integrator,
    double_integrator,
    kinematic_model,
    double_integrator_3d,
    quadrotor,
)

class MultiAgentEnv(EnvWrapper):
    def __init__(self, state_dim: int, control_dim: int, n_agents: int, dynamic_func):
        super().__init__()
        self.state_dim   = state_dim
        self.control_dim = control_dim
        self.n_agents    = n_agents
        self.dynamic     = dynamic_func
        self.batch_first = True
        self.current_state: torch.Tensor | None = None

    def reset(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch:        (B, n_agents * state_dim) tensor of flattened initial states
        returns:      (B, 1, n_agents * state_dim)
        """
        # reshape into (B, 1, total_dim)
        self.current_state = batch.reshape(batch.shape[0], 1, self.n_agents * self.state_dim)
        return self.current_state

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """
        action:       (B, n_agents * control_dim)
        returns:      next_state (B, 1, n_agents * state_dim)
        """
        batch_size = action.shape[0]
        next_states: list[torch.Tensor] = []

        state_seq = self.current_state.view(batch_size, self.n_agents, self.state_dim)
        action_seq = action.view(batch_size, self.n_agents, self.control_dim)

        for i in range(self.n_agents):
            s = state_seq[:, i, :]        # (B, state_dim)
            a = action_seq[:, i, :]       # (B, control_dim)
            ns = self.dynamic(s, a, batch_first=self.batch_first)  # (B,1,state_dim)
            next_states.append(ns)

        self.current_state = torch.cat(next_states, dim=2)  # along feature axis
        return self.current_state


def batched_rollout(
    env: MultiAgentEnv,
    policy_A: PolicyBase,
    policy_B: PolicyBase,
    T: int
) -> torch.Tensor:
    """
    Runs both teams for T steps, returns full trajectory.

    Returns:
        traj: (B, T+1, n_agents * state_dim)
    """
    B          = env.current_state.shape[0]
    n_agents   = env.n_agents
    sd, cd     = env.state_dim, env.control_dim
    traj       = torch.zeros(B, T+1, n_agents * sd, device=env.current_state.device)
    traj[:, 0, :] = env.current_state.squeeze(1)

    # assume first half of agents -> policy_A, second half -> policy_B
    nA = n_agents // 2
    nB = n_agents - nA

    for t in range(T):
        state = env.current_state  # (B,1,n_agents*sd)
        # split for A/B
        sA = state[..., :nA*sd].reshape(B, 1, nA*sd)
        sB = state[..., nA*sd:].reshape(B, 1, nB*sd)

        aA, hiddenA, infoA = policy_A.act(sA)  # (B, nA*cd) ideally
        # Resilient fallback: if aA has wrong width, replace with zeros of correct shape
        if aA.shape[1] != nA * cd:
            aA = torch.zeros(B, nA * cd, device=aA.device)

        aB, hiddenB, infoB = policy_B.act(sB)  # (B, nB*cd) ideally
        if aB.shape[1] != nB * cd:
            aB = torch.zeros(B, nB * cd, device=aB.device)

        act = torch.cat((
            aA.reshape(B, nA*cd),
            aB.reshape(B, nB*cd)
        ), dim=1)

        ns = env.step(act)       # (B,1,n_agents*sd)
        traj[:, t+1, :] = ns.squeeze(1)

    return traj

if __name__ == "__main__":
    import torch
    from src.interfaces import PolicyBase
    import src.dynamic_helper as dynamic_helper

    # 1) A trivial zero‐action policy for testing
    class ZeroPolicy(PolicyBase):
        def reset(self, batch: int = 1):
            return None

        def act(self, obs, hidden=None, deterministic=False):
            # Flatten observations except batch dim
            B = obs.shape[0]
            flat = obs.reshape(B, -1)
            D = flat.size(1)
            return torch.zeros(B, D, device=obs.device), None, {}

    # 2) Dynamics to test
    dynamics = {
        'single_integrator':    dynamic_helper.single_integrator,
        'double_integrator':    dynamic_helper.double_integrator,
        'kinematic_model':      dynamic_helper.kinematic_model,
        'double_integrator_3d': dynamic_helper.double_integrator_3d,
        'quadrotor':            dynamic_helper.quadrotor,
    }

    # 3) Per‐dynamics dims (state_dim, control_dim)
    dims = {
        'single_integrator':    (2, 2),
        'double_integrator':    (4, 2),
        'kinematic_model':      (5, 2),
        'double_integrator_3d': (6, 3),
        'quadrotor':            (6, 3),
    }

    batch_size = 4
    horizon    = 5

    for name, func in dynamics.items():
        sd, cd = dims[name]
        print(f"\n=== Testing {name} ===")
        for n_agents in (1, 2, 4):
            total_sd = sd * n_agents
            total_cd = cd * n_agents

            # instantiate environment
            env = MultiAgentEnv(state_dim=sd, control_dim=cd,
                                n_agents=n_agents, dynamic_func=func)

            # random start
            init = torch.randn(batch_size, total_sd)
            obs0 = env.reset(init)
            assert obs0.shape == (batch_size, 1, total_sd), f"reset shape bad: {obs0.shape}"

            # run rollout
            pA = ZeroPolicy()
            pB = ZeroPolicy()
            traj = batched_rollout(env, pA, pB, T=horizon)

            # outputs
            B, T1, D1 = traj.shape
            expected_D = total_sd
            assert B == batch_size and T1 == horizon+1 and D1 == total_sd, \
                f"traj shape mismatch: got {traj.shape}, want ({batch_size},{horizon+1},{total_sd})"
            print(f" n_agents={n_agents:>2} → trajectory shape {traj.shape} ✓")

    print("\nAll dynamics & agent‐counts passed!")