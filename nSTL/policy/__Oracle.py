from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from src.interfaces import OracleBase, PolicyBase, EnvWrapper
from env.dynamic_env import MultiAgentEnv, batched_rollout, differentiable_rollout
from robust.team_stl_helper import STLFormulaReachAvoidTeam
from policy.modules_helper import LSTM
from config.team_config import ConfigTeam
from policy.team_modules_helper import JointPolicy, make_joint_policy
import random
from torch.distributions import Categorical
import wandb
from tqdm import trange

MARGIN = 0.3              # robustness margin used in STL‑Game paper
REF_LOSS_W = 0.1          # weight for reference trajectory shaping
START_NOISE_STD = 0.02    # gaussian jitter for deterministic starts
K_OPP = 4                 # how many opponents to sample per epoch

def _reference_traj(batch, T, device):
    """Return a straight‑line reference to the origin (goal) with shape (B,T,2)."""
    ref = torch.zeros(batch, T, 2, device=device)
    return ref


class JointSTLOracle(OracleBase):
    """
    Uses differentiable STL robustness to train a joint best-response.
    Implements OracleBase.train(team_id, opp_meta) -> JointPolicy.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        stl_formula: STLFormulaReachAvoidTeam,
        cfg: ConfigTeam, 
        team_size: int,
        total_t: int, 
    ):
        self.env = env
        self.stl = stl_formula
        self.cfg = cfg 
        self.t = total_t
        self.epochs  = getattr(cfg, "epochs", 1)
        self.lr      = getattr(cfg, "lr", 1e-4)


        # GET Property values
        self.team_size = team_size
        self.sd = self.env.get_state_dim
        self.cd = self.env.get_control_dim

    # --------------------------------------------------------------------- #
    # rollout helper                                                        #
    # --------------------------------------------------------------------- #
    def _one_rollout(
        self,
        joint_policy: JointPolicy,
        opp_policy: LSTM,
        init_states: torch.Tensor,
    ):
        """Run a single rollout and compute STL robustness (no gradients)."""
        env = self.env
        sd = self.sd
        env.reset(init_states)
        _ = joint_policy.reset(init_states.size(0))
        _ = opp_policy.reset(init_states.size(0))

        # Differentiable rollout through dynamics
        traj = differentiable_rollout(
            env,
            joint_policy,          # policy_A (ego team)
            opp_policy,            # policy_B (opponent team)
            init_states,
            self.t,
        )
        ego_trajs = [traj[:, :, i * sd:(i + 1) * sd] for i in range(self.team_size)]
        opp_size = env.n_agents - self.team_size
        opp_trajs = [traj[:, :, (self.team_size + k) * sd:
                                   (self.team_size + k + 1) * sd]
                     for k in range(opp_size)]
        rob = self.stl.compute_robustness_ego(ego_trajs, opp_trajs)
        return traj, rob

    # --------------------------------------------------------------------- #
    # opponent selection helpers                                            #
    # --------------------------------------------------------------------- #
    def _select_worst_opponent(
        self,
        joint_policy: JointPolicy,
        opp_meta: List[LSTM],
        init_states: torch.Tensor,
    ) -> LSTM:
        """
        Return the opponent policy from `opp_meta` that minimises STL robustness
        against `joint_policy` for the provided initial states.

        Args
        ----
        joint_policy : the current (gradient‑bearing) ego joint policy
        opp_meta     : list of opponent policies in the meta‑population
        init_states  : tensor (B, n_agents * state_dim) of starting states

        Notes
        -----
        * Runs without gradient tracking so the overhead is just forward passes.
        * Each opponent is evaluated on the SAME initial states to keep the
          comparison fair.
        """
        env = self.env
        sd = self.sd
        worst_pol = opp_meta[0]
        worst_val = float("inf")

        with torch.no_grad():
            for pol in opp_meta:
                # reset env and both policies
                env.reset(init_states)
                _ = joint_policy.reset(init_states.size(0))
                _ = pol.reset(init_states.size(0))

                traj = differentiable_rollout(
                    env,
                    joint_policy,      # ego
                    pol,               # candidate opponent
                    init_states,
                    self.t,
                )

                # slice trajectories into ego/opp positional traces
                ego_trajs = [traj[:, :, i * sd:(i + 1) * sd]                      # ego agents
                             for i in range(self.team_size)]
                opp_size = env.n_agents - self.team_size
                opp_trajs = [traj[:, :, (self.team_size + k) * sd:
                                          (self.team_size + k + 1) * sd]
                             for k in range(opp_size)]

                rob = self.stl.compute_robustness_ego(ego_trajs, opp_trajs)
                val = rob.mean().item()
                if val < worst_val:
                    worst_val = val
                    worst_pol = pol
        return worst_pol

    def train(
            self,
            team_id: str,
            opp_meta: List[LSTM],
            mu_opp: torch.Tensor = None,
            payoff_queue=None,
            driver=None,
            flush_every: int = 50,
    ) -> JointPolicy:
        """
        Train a joint best‑response.

        Parameters
        ----------
        team_id : "A" or "B"
        opp_meta : list of opponent joint policies
        mu_opp : tensor of shape (len(opp_meta),) – meta‑strategy of the opponent
                 (if None, uniform distribution is used)
        payoff_queue : optional queue to push (self_idx, opp_idx, robustness)
        driver : optional PSRODriver; if provided we call driver.flush_payoff_queue()
                 every `flush_every` epochs and refresh mu_opp from driver.mu_B / mu_A.
        flush_every : how often (epochs) to flush and refresh
        """
        device = self.cfg.device
        if mu_opp is None:
            mu_opp = torch.ones(len(opp_meta), device=device, dtype=torch.float32) / len(opp_meta)
        else:
            # make sure mu_opp is a torch tensor on the correct device
            if not torch.is_tensor(mu_opp):
                mu_opp = torch.tensor(mu_opp, device=device, dtype=torch.float32)
            else:
                mu_opp = mu_opp.to(device=device, dtype=torch.float32)
        # ensure it sums to 1 to avoid numerical issues
        mu_opp = mu_opp / mu_opp.sum()

        dist = Categorical(mu_opp)

        # build new joint policy
        joint_policy = make_joint_policy(
            team_size=self.team_size,
            sd=self.sd,
            cd=self.cd,
            device=device,
            TOTAL_T=self.t,
            team_id=team_id,
        )
        optimizer = torch.optim.Adam(joint_policy.parameters(), self.lr)
        print(f"[Oracle] Constructed Joint Policy for Team {team_id}")

        B = self.cfg.batch_size
        total_sd = self.env.n_agents * self.sd

        for epoch in trange(self.epochs, desc=f"Oracle {team_id}", leave=False):
            losses = []
            rob_collect = []

            for _ in range(K_OPP):
                # sample opponent index according to meta‑mix
                opp_idx = dist.sample().item()
                opp_policy = opp_meta[opp_idx]

                # fixed deterministic start + small noise
                init = torch.zeros(B, total_sd, device=device)
                init[:, 0:2] = torch.tensor([-1.0, -1.0], device=device)
                init[:, 2:4] = torch.tensor([-1.0, -1.0], device=device)
                init[:, 4:6] = torch.tensor([1.0, 1.0], device=device)
                init[:, 6:8] = torch.tensor([1.0, 1.0], device=device)
                init += START_NOISE_STD * torch.randn_like(init)

                if driver is not None:
                    driver._last_init = init.detach().clone()

                traj, rho = self._one_rollout(joint_policy, opp_policy, init)

                # extract ego trajectory (assumes first 2 dims are x,y)
                ego_traj = traj[:, :, :2]
                # slice ego trajectory for ref‑loss
                ref_traj = _reference_traj(B, ego_traj.size(1), device)  # now also T+1

                # STL margin reward
                reward = -F.relu(-rho + MARGIN)

                # flip sign for opponent team
                if team_id.upper() == "B":
                    reward = -reward

                # reference shaping loss (MSE)
                ref_loss = F.mse_loss(ego_traj, ref_traj)

                # entropy
                ent = joint_policy.entropy().mean() if hasattr(joint_policy, "entropy") else 0.0

                loss_k = -(reward.mean() - REF_LOSS_W * ref_loss - 0.01 * ent)
                losses.append(loss_k)
                rob_collect.append(rho.detach())

            loss = torch.stack(losses).mean()
            rob_batch = torch.stack(rob_collect).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % flush_every == 0:
                wandb.log({
                    f"{team_id}/oracle_loss": loss.item(),
                    f"{team_id}/robustness": rob_batch.item(),
                    f"{team_id}/epoch": epoch + 1
                })

                if driver is not None:
                    driver.flush_payoff_queue()
                    new_mix = torch.tensor(
                        driver.mu_B if team_id.upper() == "A" else driver.mu_A,
                        device=device,
                    )
                    dist = Categorical(new_mix)

        return joint_policy

    def __repr__(self):
        return f"<JointSTLOracle team_size={self.team_size}>"
