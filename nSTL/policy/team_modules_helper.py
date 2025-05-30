"""
A JointPolicy: 

“team” of n individual agent-policies as one coherent policy when you do rollouts or oracle training. 

	1.	Holds a list of n AgentPolicy objects 
	2.	Splits the joint observation tensor into n per agent observations
	3.	Calls each agent’s .act(forward) -> stochastic/deterministic
    4.  Concat all actions to one big action
	5.	Stacks the new hidden states

"""
from typing import List, Optional, Tuple, Dict
import torch
from torch import Tensor
import torch.nn as nn
from src.interfaces import PolicyBase
from policy.modules_helper import LSTM, RecurrentAgent


class JointPolicy(PolicyBase, nn.Module):
    def __init__(self, agent_policies: List[LSTM], team_id: str):
        """
        agent_policies: list of length n_agents,
                        each maps obs_i -> (action_i, hidden_i, info_i)
        """
        super().__init__()
        self.policies = nn.ModuleList(agent_policies)
        # assume all agents use the same hidden‐state structure/size
        self.n_agents = len(agent_policies)
        self.team_id = team_id

    def __name__(self):
        return self.team_id

    def reset(self, batch_size: int = 1):
        """
        Call .reset() on each sub‐policy and collect their initial hidden states.
        """
        self.hiddens = []
        for p in self.policies:
            h = p.reset(batch_size)
            self.hiddens.append(h)
        return self.hiddens

    def act(
        self,
        joint_obs: Tensor,              # shape (B, 1, n_agents * state_dim)
        hidden: Optional[List] = None,   # list of per-agent hidden states
        deterministic: bool = False
    ) -> Tuple[Tensor, List, Dict]:
        B, _, total_sd = joint_obs.shape
        sd_per_agent = total_sd // self.n_agents

        # split obs for each agent along feature axis:
        obs_slices = torch.split(joint_obs, sd_per_agent, dim=2)

        actions = []
        new_hiddens = []
        infos = {}

        for idx, p in enumerate(self.policies):
            obs_i = obs_slices[idx]            # shape (B,1,sd_per_agent)
            hid_i = hidden[idx] if hidden else None
            # each policy returns (action_i, hidden_i_new, info_i)
            a_i, info_i = p.act(obs_i, hid_i, deterministic=deterministic)
            actions.append(a_i)                     # expect (B, action_dim)
            # optionally merge info dicts under keys 'agent0', 'agent1', …
            infos[f"agent{idx}"] = info_i

        # concatenate per-agent actions along feature axis
        joint_action = torch.cat(actions, dim=1)     # shape (B, n_agents * action_dim)

        return joint_action, new_hiddens, infos



def make_joint_policy(team_size, sd, cd, TOTAL_T, device, team_id):
    """
    Batch function to quickly build Team of LSTM agents
    # TODO: Extend to Reccurrent Agent??
    """
    agents = [
        LSTM(
            hidden_dim=64,
            pred_length=TOTAL_T,
            state_dim = sd,
            action_dim= cd,
            stochastic_policy=False,
            batch_first=True,
            action_max=1.0
        )
        for _ in range(team_size)
    ]
    jp = JointPolicy(agents, team_id)
    return jp