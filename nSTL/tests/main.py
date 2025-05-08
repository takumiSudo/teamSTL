"""Main entry point for running Team PSRO experiments.

Usage examples
--------------
python main.py                              # default single integrator 2 x 2 game
python main.py --dyn kinematic_model        # choose a different dynamics
python main.py --iters 15 --epochs 400       # more PSRO iterations / oracle epochs

All artifacts (WANDB logs + GIF) are saved under `artifact/figs/results/main/`.
"""
import os, argparse, wandb, torch, matplotlib.pyplot as plt
import shutil
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

def animate(ego_trajs, opp_trajs, obstacles, circle, save_path, interval_ms=200):
    """Render the two‑team rollout with fading trails and start/goal markers."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # If a directory with the same name as the target GIF exists from a failed run,
    # remove it so Pillow can create a file at that path.
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    elif os.path.isfile(save_path):
        os.remove(save_path)

    T = ego_trajs[0].shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Team‑PSRO Roll‑out")
    ax.grid(ls=":", lw=0.5)

    # obstacles
    obs1, obs2 = obstacles[:2]
    for obs, c in zip((obs1, obs2), ("#d62728", "#2ca02c")):
        xlo, xhi, ylo, yhi = obs
        ax.fill([xlo, xhi, xhi, xlo], [ylo, ylo, yhi, yhi], alpha=0.3, color=c)

    # circle region
    cx, cy, r = circle
    ang = torch.linspace(0, 2 * torch.pi, 200)
    ax.plot(cx + r * torch.cos(ang), cy + r * torch.sin(ang), "#1f77b4", lw=1.5)

    # draw fading trails (collections of Line2D)
    trails_ego = [ax.plot([], [], color="#ff7f0e", alpha=float(a))[0] for a in torch.linspace(0.2, 1.0, 10)]
    trails_opp = [ax.plot([], [], color="#9467bd", alpha=float(a))[0] for a in torch.linspace(0.2, 1.0, 10)]

    head_ego, = ax.plot([], [], "o", c="#ff7f0e", ms=6, label="ego team")
    head_opp, = ax.plot([], [], "o", c="#9467bd", ms=6, label="opp team")
    ax.legend(loc="upper right")

    def update(t):
        # tails of length 10
        for k in range(10):
            idx = max(t - k, 0)
            e_xy = torch.vstack([tr[idx] for tr in ego_trajs]).detach().cpu()
            o_xy = torch.vstack([tr[idx] for tr in opp_trajs]).detach().cpu()
            trails_ego[k].set_data(e_xy[:, 0], e_xy[:, 1])
            trails_opp[k].set_data(o_xy[:, 0], o_xy[:, 1])
        # heads
        head_ego.set_data(e_xy[:, 0], e_xy[:, 1])
        head_opp.set_data(o_xy[:, 0], o_xy[:, 1])
        return trails_ego + trails_opp + [head_ego, head_opp]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=True)
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Animation saved → {save_path}")

# -------------------------------------------------------------
# main routine
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Team‑PSRO runner")
    parser.add_argument("--dyn", default="single_integrator",
                        choices=["single_integrator", "double_integrator", "kinematic_model", "double_integrator_3d", "quadrotor"],
                        help="dynamics model")
    parser.add_argument("--iters", type=int, default=10, help="PSRO iterations")
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
    ego_start = [-1.0, -1.0]
    opp_start = [0.5, 0.5]
    cfg = ConfigTeam(ego_start, opp_start)
    cfg.fsp_iteration = args.iters
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch
    cfg.lr = args.lr

    # wandb run
    wandb.init(project=f"team-stl-psro_{args.dyn}", name=f"{args.dyn}_main_{cfg.T}", config=vars(args))
    best_robust = float("-inf")
    best_ego, best_opp = None, None

    driver = PSRODriver(cfg, dyn_fn, state_dim, ctrl_dim, team_size=2)
    for it in range(args.iters):
        exp = driver.iterate()
        wandb.log({"iteration": it, "exploitability": -exp})
        if it % 10 == 0:
            T = cfg.T
            traj = batched_rollout(driver.env, driver.pop_A[-1], driver.pop_B[-1], T=T)
            sd = driver.env.state_dim
            pos_dim = 2
            ego = [traj[:, :, i*sd:(i*sd)+pos_dim] for i in range(2)]
            opp = [traj[:, :, (2+i)*sd:(2+i)*sd+pos_dim] for i in range(2)]
            gif_path = os.path.join("artifact/figs/results/main", f"{args.dyn}_psro_{it}.gif")
            animate(ego, opp, driver.cfg.get_obstacles(), driver.cfg.get_obstacles()[2], gif_path)
            wandb.save(gif_path)
            # track the best‑scoring trajectory so far
            rob_it = driver.stl.compute_robustness_ego(ego, opp)
            if rob_it > best_robust:
                best_robust = rob_it
                best_ego = [tr.clone() for tr in ego]
                best_opp = [tr.clone() for tr in opp]

    # visualise the best trajectory obtained during training
    if best_ego is not None:
        best_gif_path = os.path.join("artifact/figs/results/main", f"{args.dyn}_best.gif")
        animate(best_ego, best_opp, driver.cfg.get_obstacles(),
                driver.cfg.get_obstacles()[2], best_gif_path, interval_ms=300)
        wandb.save(best_gif_path)

    # final rollout for visualisation
    T = 50
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
    animate(ego, opp, driver.cfg.get_obstacles(), driver.cfg.get_obstacles()[2],
            gif_path, interval_ms=300)
    wandb.save(gif_path)
    wandb.finish()


if __name__ == "__main__":
    main()
