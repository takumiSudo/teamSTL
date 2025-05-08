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
        self.lr      = getattr(cfg, "lr", 1e-3)


        # GET Property values
        self.team_size = team_size
        self.sd = self.env.get_state_dim
        self.cd = self.env.get_control_dim

    def train(
            self, 
            team_id: str, 
            opp_meta: List[LSTM]
    )-> JointPolicy:
        joint_policy = make_joint_policy(team_size=self.team_size, sd=self.sd, cd=self.cd, device=self.cfg.device, TOTAL_T=self.t, team_id=team_id)
        optimizer = torch.optim.Adam(joint_policy.parameters(), self.lr)
        print(f"Constructed Joint Policy for Team {team_id}")
        print(joint_policy)

        for _ in range(self.epochs):
            env = self.env
            # initialize full state for all agents (both teams)
            B = self.cfg.batch_size
            total_sd = env.n_agents * self.sd
            opp_policy = random.choice(opp_meta)
            # sample random initial positions uniformly in [-1,1]
            init = (torch.rand(B, total_sd, device=self.cfg.device) * 2.0 - 1.0).requires_grad_(True)
            sd = self.sd
            env.reset(init)

            h_team = joint_policy.reset(B)
            _ = opp_policy.reset(B)

            traj = batched_rollout(env, joint_policy, opp_policy, self.t)
            ego_trajs = [traj[:, :, i*sd:(i+1)*sd] for i in range(self.team_size)]
            opp_size  = env.n_agents - self.team_size
            opp_trajs = [traj[:, :, (self.team_size+i)*sd:(self.team_size+i+1)*sd]
                         for i in range(opp_size)]


            # compute differentiable STL robustness
            print("ego_trajs:", [t.shape for t in ego_trajs], ego_trajs[0])
            print("opp_trajs:", [t.shape for t in opp_trajs], opp_trajs[0])
            rob = self.stl.compute_robustness_ego(ego_trajs, opp_trajs)
            loss = -rob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return joint_policy

    def __repr__(self):
        return f"<JointSTLOracle team_size={self.team_size}>"
