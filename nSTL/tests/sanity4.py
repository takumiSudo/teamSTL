import os
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from src.interfaces import EnvWrapper
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
from policy.modules_helper import LSTM
from policy.team_modules_helper import JointPolicy, make_joint_policy
from policy.Oracle import JointSTLOracle  


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
        ego_xy = torch.vstack([traj[t] for traj in ego_trajs]).detach().cpu().numpy()
        opp_xy = torch.vstack([traj[t] for traj in opp_trajs]).detach().cpu().numpy()
        ego_line.set_data(ego_xy[:, 0], ego_xy[:, 1])
        opp_line.set_data(opp_xy[:, 0], opp_xy[:, 1])
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


if __name__ == "__main__":
    TOTAL_T = 50
    SAFE_D  = torch.tensor(0.25)
    SAVE_DIR = "artifact/figs/team/env_tests/sanity_4/"

    # config & obstacles
    cfg    = ConfigTeam([], [])
    cfg.epochs = 200
    OBS    = cfg.get_obstacles()
    CIRCLE = OBS[2]
    GOAL   = OBS[4]

    # dynamics to test
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
        multi_env = MultiAgentEnv(
            state_dim=sd, control_dim=cd,
            n_agents=n_agents, dynamic_func=func
        )

        # random init
        B    = 1
        init = torch.rand(B, n_agents * sd) * 2.0 - 1.0
        multi_env.reset(init)

        # build initial untrained JointPolicy for both teams
        policy_A = make_joint_policy(2, sd, cd, TOTAL_T, cfg.device, "A")
        policy_B = make_joint_policy(2, sd, cd, TOTAL_T, cfg.device, "B")
        policy_A.reset(batch_size=B)
        policy_B.reset(batch_size=B)

        # instantiate and run gradient oracle to train A against B
        stl = STLFormulaReachAvoidTeam(
            obs_boxes=[OBS[0], OBS[1]],
            circle_obs=CIRCLE,
            goal_box=GOAL,
            T=cfg.T,
            safe_d=SAFE_D
        )
        oracle = JointSTLOracle(
            env=multi_env,
            stl_formula=stl,
            cfg=cfg,
            team_size=2,
            total_t=cfg.T
        )
        print("Training joint best-response for team A...")
        trained_A = oracle.train(team_id="A", opp_meta=[policy_B])

        # rollout with trained squad
        traj = batched_rollout(multi_env, trained_A, policy_B, T=TOTAL_T)
        # keep batch dimension when splitting trajectories
        # ego_trajs = [traj[:, :, i*sd:(i+1)*sd] for i in range(2)]
        # opp_trajs = [traj[:, :, (2+i)*sd:(2+i+1)*sd] for i in range(2)]
        pos_dim = 2
        ego_trajs = [traj[:, :, i*sd + 0 : i*sd + pos_dim] for i in range(2)]
        opp_trajs = [traj[:, :, (2+i)*sd + 0 : (2+i)*sd + pos_dim] for i in range(2)]

        # compute robustness
        rob_ego = stl.compute_robustness_ego(ego_trajs, opp_trajs)
        rob_opp = stl.compute_robustness_opp(ego_trajs, opp_trajs)
        print(f"After training → Ego robustness: {rob_ego.item():.3f}, Opp robustness: {rob_opp.item():.3f}")

        # animate
        animate_scene(
            ego_trajs, opp_trajs, OBS, CIRCLE,
            fname=f"{name}.gif", SAVE_DIR=SAVE_DIR
        )
