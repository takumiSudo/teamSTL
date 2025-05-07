"""Main entry point for running Team‑PSRO experiments.

Usage examples
--------------
python main.py                              # default single‑integrator 2×2 game
python main.py --dyn kinematic_model        # choose a different dynamics
python main.py --iters 15 --epochs 400       # more PSRO iterations / oracle epochs

All artifacts (WANDB logs + GIF) are saved under `artifact/figs/results/main/`.
"""
import os, argparse, wandb, torch, matplotlib.pyplot as plt
from matplotlib import animation

from config.team_config import ConfigTeam
from env.dynamic_env import batched_rollout
from env.dynamic_helper import (
    single_integrator,
    double_integrator,
    kinematic_model,
    double_integrator_3d,
    quadrotor,
)
from policy.PSRO import PSRODriver

def animate(ego_trajs, opp_trajs, obstacles, circle, fname, SAVE_DIR):
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

# -------------------------------------------------------------
# main routine
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Team‑PSRO runner")
    parser.add_argument("--dyn", default="single_integrator",
                        choices=["single_integrator", "double_integrator", "kinematic_model", "double_integrator_3d", "quadrotor"],
                        help="dynamics model")
    parser.add_argument("--iters", type=int, default=100, help="PSRO iterations")
    parser.add_argument("--epochs", type=int, default=200, help="oracle training epochs")
    parser.add_argument("--batch", type=int, default=256, help="oracle RL batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="oracle learning rate")
    args = parser.parse_args()

    dyn_map = {
        "single_integrator":    (single_integrator, 2, 2),
        "double_integrator":    (double_integrator, 4, 2),
        "kinematic_model":      (kinematic_model, 5, 2),
        "double_integrator_3d": (double_integrator_3d, 6, 3),
        "quadrotor":            (quadrotor, 6, 3),
    }
    dyn_fn, state_dim, ctrl_dim = dyn_map[args.dyn]

    # basic config
    cfg = ConfigTeam([], [])
    cfg.fsp_iteration = args.iters
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch
    cfg.lr = args.lr

    # wandb run
    wandb.init(project="team-stl-psro", name=f"{args.dyn}_main", config=vars(args))

    driver = PSRODriver(cfg, dyn_fn, state_dim, ctrl_dim, team_size=2)
    for it in range(args.iters):
        exp = driver.iterate()
        wandb.log({"iteration": it, "exploitability": -exp})

    # final rollout for visualisation
    T = cfg.T
    traj = batched_rollout(driver.env, driver.pop_A[-1], driver.pop_B[-1], T=T)
    sd = driver.env.state_dim
    pos_dim = 2
    ego = [traj[:, :, i*sd:(i*sd)+pos_dim] for i in range(2)]
    opp = [traj[:, :, (2+i)*sd:(2+i)*sd+pos_dim] for i in range(2)]

    rob_ego = driver.stl.compute_robustness_ego(ego, opp)
    rob_opp = driver.stl.compute_robustness_opp(ego, opp)
    wandb.log({"robustness_ego": rob_ego.item(), "robustness_opp": rob_opp.item()})

    save_dir = "artifact/figs/results/main"
    gif_path = os.path.join(save_dir, f"{args.dyn}_psro.gif")
    animate(ego, opp, driver.cfg.get_obstacles(), driver.cfg.get_obstacles()[2], gif_path)
    wandb.save(gif_path)
    wandb.finish()


if __name__ == "__main__":
    main()
