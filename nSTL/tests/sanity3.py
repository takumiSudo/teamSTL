# sanity_full.py

import os
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from env.dynamic_env import MultiAgentEnv, batched_rollout
from robust.team_stl_helper import STLFormulaReachAvoidTeam
from config.team_config import ConfigTeam
from env.dynamic_helper import (
    single_integrator,
    double_integrator,
    kinematic_model,
    double_integrator_3d,
    quadrotor,
)

# ----------------- helper: linear interp --------------------
def line_traj(start_xy, goal_xy, T):
    xs = torch.linspace(start_xy[0], goal_xy[0], T).unsqueeze(1)
    ys = torch.linspace(start_xy[1], goal_xy[1], T).unsqueeze(1)
    return torch.cat((xs, ys), dim=1)

# ----------------- helper: animate -------------------------
def animate_scene(ego_trajs, opp_trajs, obstacles, circle, fname, SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.set_aspect("equal"); ax.grid()

    # plot obstacles once
    obs1, obs2 = obstacles[:2]
    for obs, color in zip((obs1, obs2), ("r","g")):
        xlo, xhi, ylo, yhi = obs
        ax.plot([xlo,xlo,xhi,xhi,xlo],[ylo,yhi,yhi,ylo,ylo], color)

    # plot circle
    cx, cy, rad = circle
    ang = torch.linspace(0,2*torch.pi,100)
    ax.plot(cx + rad*torch.cos(ang),
            cy + rad*torch.sin(ang), "b")

    ego_line, = ax.plot([], [], '-o', c='orange', label='ego')
    opp_line, = ax.plot([], [], '-o', c='purple', label='opp')
    ax.legend()

    def init():
        ego_line.set_data([], []); opp_line.set_data([], [])
        return ego_line, opp_line

    def animate(t):
        ego_xy = torch.vstack([traj[t] for traj in ego_trajs])
        opp_xy = torch.vstack([traj[t] for traj in opp_trajs])
        ego_line.set_data(ego_xy[:,0], ego_xy[:,1])
        opp_line.set_data(opp_xy[:,0], opp_xy[:,1])
        return ego_line, opp_line

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=ego_trajs[0].shape[0],
        interval=80, blit=True
    )
    path = os.path.join(SAVE_DIR, fname)
    anim.save(path, writer="pillow")
    plt.close(fig)
    print(f"Saved animation: {path}")

# ----------------- zero-action policy -----------------------
from policy.modules_helper import LSTM
from policy.team_modules_helper import JointPolicy

# suppose 2 agents per team, each partial‐policy is an LSTM returning (action, hidden, {})
agent_policies = [ LSTM(input_dim=sd, hidden_dim=64, output_dim=cd)
                 for _ in range(2) ]
team_policy = JointPolicy(agent_policies)

# in your rollout:
h0 = team_policy.reset(batch_size=B)
action, h1, _ = team_policy.act(joint_obs, h0)


if __name__ == "__main__":
    TOTAL_T = 50
    SAFE_D  = torch.tensor(0.25)
    SAVE_DIR = "artifact/figs/team/env_tests/"

    # reuse obstacles & circle from config
    cfg    = ConfigTeam([], [])
    OBS    = cfg.get_obstacles()
    CIRCLE = OBS[2]
    GOAL   = OBS[4]

    # dynamics to test and their dims (state_dim, control_dim)
    dynamics = {
        'single_integrator':    (single_integrator,    (2,2)),
        'double_integrator':    (double_integrator,    (4,2)),
        'kinematic_model':      (kinematic_model,      (5,2)),
        'double_integrator_3d': (double_integrator_3d, (6,3)),
        'quadrotor':            (quadrotor,            (6,3)),
    }

    for name, (func, (sd, cd)) in dynamics.items():
        print(f"\n=== {name} ===")
        n_agents = 4  # 2v2
        env = MultiAgentEnv(
            state_dim=sd, control_dim=cd,
            n_agents=n_agents, dynamic_func=func
        )

        # random initial states
        B    = 1
        init = torch.rand(B, n_agents * sd) * 2.0 - 1.0   # in [−1,1]
        _    = env.reset(init)

        assert ((init >= -1.0) & (init <= 1.0)).all(), \
            f"Initial states out of bounds: min={init.min().item():.3f}, max={init.max().item():.3f}"

        # rollout with zero-policy
        # joint policies for 2 agents per team
        agent_policies_A = [
            LSTM(input_dim=sd, hidden_dim=64, output_dim=cd)
            for _ in range(2)
        ]
        policy_A = JointPolicy(agent_policies_A)
        h0_A = policy_A.reset(batch_size=B)

        agent_policies_B = [
            LSTM(input_dim=sd, hidden_dim=64, output_dim=cd)
            for _ in range(2)
        ]
        policy_B = JointPolicy(agent_policies_B)
        h0_B = policy_B.reset(batch_size=B)

        traj = batched_rollout(env, policy_A, policy_B, T=TOTAL_T)
        # traj shape: (B, TOTAL_T+1, n_agents*sd)

        # extract x,y for each of 2 egos & 2 opps
        full      = traj[0]  # (TOTAL_T+1, n_agents*sd)
        ego_trajs = [full[:, i*sd:(i*sd+2)] for i in range(2)]
        opp_trajs = [full[:, (2+i)*sd:(2+i)*sd+2] for i in range(2)]

        # compute robustness
        stl = STLFormulaReachAvoidTeam(
            obs_boxes=[OBS[0], OBS[1]],
            circle_obs=CIRCLE,
            goal_box=GOAL,
            T=cfg.T,
            safe_d=SAFE_D
        )
        rob_ego = stl.compute_robustness_ego(ego_trajs, opp_trajs)
        rob_opp = stl.compute_robustness_opp(ego_trajs, opp_trajs)
        print(f"Ego robustness: {rob_ego.item(): .3f}, Opp robustness: {rob_opp.item(): .3f}")

        # animate & save
        animate_scene(
            ego_trajs, opp_trajs, OBS, CIRCLE,
            fname=f"{name}.gif", SAVE_DIR=SAVE_DIR
        )