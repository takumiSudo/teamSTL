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
from policy.PSRO import PSRODriver, ZeroSumMetaSolver
import wandb


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
    SAVE_DIR = "artifact/figs/team/env_tests/sanity_5/"

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
        # initialize wandb for this dynamics
        wandb.init(
            project="team-stl-psro",
            name=name,
            config={
                'dynamics': name,
                'fsp_iterations': cfg.fsp_iteration,
                'epochs': cfg.epochs,
                'batch_size': cfg.batch_size,
                'lr': cfg.lr
            }
        )
        # full PSRO loop
        driver = PSRODriver(cfg, func, sd, cd, team_size=2)
        for itr in range(cfg.fsp_iteration):
            exploit = driver.iterate()
            wandb.log({'iteration': itr, 'exploitability': exploit})
            
        # final rollout & compute robustness
        final_env = driver.env
        policy_A = driver.pop_A[-1]
        policy_B = driver.pop_B[-1]
        traj = batched_rollout(final_env, policy_A, policy_B, T=TOTAL_T)
        sd_dim = final_env.state_dim
        pos_dim = 2
        ego_trajs = [traj[:, :, i*sd_dim:(i*sd_dim+pos_dim)] for i in range(2)]
        opp_trajs = [traj[:, :, (2+i)*sd_dim:(2+i)*sd_dim+pos_dim] for i in range(2)]
        rob_ego = driver.stl.compute_robustness_ego(ego_trajs, opp_trajs)
        rob_opp = driver.stl.compute_robustness_opp(ego_trajs, opp_trajs)
        wandb.log({'final robustness_ego': rob_ego.item(), 'final robustness_opp': rob_opp.item()})
        print(f"After PSRO → Ego robustness: {rob_ego.item():.3f}, Opp robustness: {rob_opp.item():.3f}")
        wandb.finish()

        animate_scene(
            ego_trajs, opp_trajs, OBS, CIRCLE,
            fname=f"{name}_psro.gif", SAVE_DIR=SAVE_DIR
        )
