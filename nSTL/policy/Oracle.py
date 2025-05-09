from typing import List
import torch
from torch import nn
from src.interfaces import OracleBase, PolicyBase, EnvWrapper
from env.dynamic_env import MultiAgentEnv, batched_rollout
from robust.team_stl_helper import STLFormulaReachAvoidTeam
from policy.modules_helper import LSTM
from config.team_config import ConfigTeam
from policy.team_modules_helper import JointPolicy, make_joint_policy
import random
from torch.distributions import Categorical
import wandb
from tqdm import trange


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

        traj, acts = batched_rollout(env, joint_policy, opp_policy, self.t)
        ego_trajs = [traj[:, :, i * sd:(i + 1) * sd] for i in range(self.team_size)]
        opp_size = env.n_agents - self.team_size
        opp_trajs = [traj[:, :, (self.team_size + k) * sd:
                                   (self.team_size + k + 1) * sd]
                     for k in range(opp_size)]
        rob = self.stl.compute_robustness_ego(ego_trajs, opp_trajs)
        return traj, acts, rob

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

                traj = batched_rollout(env, joint_policy, pol, self.t)

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
            prev_policy = None,
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
        # ---------- hyper-parameters ---------------------------------------------
        margin      = getattr(self.cfg, "margin", 0.3)
        ref_weight  = getattr(self.cfg, "ref_loss_weight", 1.0)
        ctrl_weight = getattr(self.cfg, "ctrl_loss_weight", 0.0)
        K_opp       = getattr(self.cfg, "k_opponents", 4)          # rollouts / epoch
        goal_center = torch.tensor([0.0, 0.0], device=device)      # (cx, cy)
        # -------------------------------------------------------------------------
        B = self.cfg.batch_size
        total_sd = self.env.n_agents * self.sd

 
        for epoch in trange(self.epochs, desc=f"Oracle {team_id}", leave=False):
            total_loss = 0.0
            total_reward = 0.0

            # draw K opponents from the meta-mix
            for opp_idx in dist.sample((K_opp,)):
                opp_policy = opp_meta[opp_idx]

                # ---- deterministic (but slightly jittered) start -----------------
                init = torch.zeros(B, total_sd, device=device)
                init[:, 0:2] = torch.tensor([-1.0, -1.0], device=device)
                init[:, 2:4] = torch.tensor([-1.0, -1.0], device=device)
                init[:, 4:6] = torch.tensor([ 1.0,  1.0], device=device)
                init[:, 6:8] = torch.tensor([ 1.0,  1.0], device=device)
                init += 0.02 * torch.randn_like(init)        # tiny exploration noise
                init = init.requires_grad_(True)

                if driver is not None:
                    driver._last_init = init.detach().clone()

                # rollout – expect (traj, u); if your helper returns only traj,
                # change the line to `traj, u = self._one_rollout(...)[0], 0.0`
                traj, u, rob = self._one_rollout(joint_policy, opp_policy, init)

                # slice ego trajectories (positions only, first 2 dims)
                ego_xy = traj[..., :2]           # shape (B, T, 2)

                # ---------------------- loss terms --------------------------------
                # 1) margin-shaped robustness reward
                if team_id.upper() == "B":
                    rob = -rob                  # opponent minimises robustness
                reward = -torch.relu(-rob + margin)
                loss_rob = -reward.mean()

                # 2) reference trajectory: pull to goal centre
                ref_traj = goal_center.view(1, 1, 2).expand_as(ego_xy)
                loss_ref = torch.nn.functional.mse_loss(ego_xy, ref_traj)

                # 3) control-effort penalty (L2 norm)
                loss_ctrl = u.norm()

                # 4) entropy bonus
                entropy = joint_policy.entropy().mean() if hasattr(joint_policy, "entropy") else 0.0

                loss = (
                    loss_rob
                    + ref_weight  * loss_ref
                    + ctrl_weight * loss_ctrl
                    - 0.01        * entropy
                )

                total_loss   += loss
                total_reward += reward.mean()

                # optional: push payoff sample for this opponent
                if payoff_queue is not None:
                    payoff_queue.put((team_id, int(opp_idx), reward.detach().mean().item()))

            # average over K opponents
            total_loss /= K_opp
            total_reward /= K_opp

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # periodic logging / μ refresh
            if (epoch + 1) % flush_every == 0:
                wandb.log({
                    f"{team_id}/oracle_loss": total_loss.item(),
                    f"{team_id}/reward":      total_reward.item(),
                    f"{team_id}/epoch":       epoch + 1
                })
                if driver is not None and hasattr(driver, "flush_payoff_queue"):
                    driver.flush_payoff_queue()
                    new_mix = torch.tensor(
                        driver.mu_B if team_id == "A" else driver.mu_A, device=device
                    )
                    dist = Categorical(new_mix)

        return joint_policy

    def __repr__(self):
        return f"<JointSTLOracle team_size={self.team_size}>"
