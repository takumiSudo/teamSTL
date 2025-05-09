"""Main entry point for running Team‑PSRO experiments (improved edition).

Key upgrades
------------
✓ tqdm progress bars for immediate ETA feedback  
✓ Adaptive GIF cadence: dense early, sparse later  
✓ Log robustness of *newest* best‑response pair each iteration  
✓ Flush payoff queue so heat‑map stays fresh  
✓ Randomised start positions to prevent overfitting  
✓ Per‑dynamics sensible default hyper‑params (can still override via CLI)

Run examples
------------
python main.py                                    # default (single‑integrator)
python main.py --dyn quadrotor --iters 50         # harder dynamics
python main.py --epochs 800 --batch 512           # override defaults
"""
import os
import argparse
import shutil
import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import trange
import wandb

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

# -----------------------------------------------------------------------------#
# visualiser                                                                   #
# -----------------------------------------------------------------------------#
def animate(ego_trajs, opp_trajs, obstacles, circle, save_path, interval_ms=200):
    """Render the two‑team rollout with fading trails and start/goal markers."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # If a directory with the same name as the target GIF exists from a failed run,
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

    line_ego, = ax.plot([], [], color="#ff7f0e", lw=2, label="ego path")
    line_opp, = ax.plot([], [], color="#9467bd", lw=2, label="opp path")
    head_ego, = ax.plot([], [], "o", c="#ff7f0e", ms=6)
    head_opp, = ax.plot([], [], "o", c="#9467bd", ms=6)
    ax.legend(loc="upper right")

    def update(t):
        e_xy = torch.vstack([tr[: t + 1] for tr in ego_trajs]).cpu()
        o_xy = torch.vstack([tr[: t + 1] for tr in opp_trajs]).cpu()
        line_ego.set_data(e_xy[:, 0], e_xy[:, 1])
        line_opp.set_data(o_xy[:, 0], o_xy[:, 1])
        head_ego.set_data(e_xy[-1, 0], e_xy[-1, 1])
        head_opp.set_data(o_xy[-1, 0], o_xy[-1, 1])
        return [line_ego, line_opp, head_ego, head_opp]

    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=interval_ms, blit=True)
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Animation saved → {save_path}")

# -----------------------------------------------------------------------------#
# defaults per dynamics                                                        #
# -----------------------------------------------------------------------------#
DYN_TABLE = {
    "single_integrator":    {"fn": single_integrator,    "state_dim": 2, "ctrl_dim": 2,
                             "lr": 3e-4, "batch": 256, "epochs": 400},
    "double_integrator":    {"fn": double_integrator,    "state_dim": 4, "ctrl_dim": 2,
                             "lr": 3e-4, "batch": 512, "epochs": 600},
    "kinematic_model":      {"fn": kinematic_model,      "state_dim": 5, "ctrl_dim": 2,
                             "lr": 3e-4, "batch": 512, "epochs": 600},
    "double_integrator_3d": {"fn": double_integrator_3d, "state_dim": 6, "ctrl_dim": 3,
                             "lr": 5e-4, "batch": 1024, "epochs": 800},
    "quadrotor":            {"fn": quadrotor,            "state_dim": 6, "ctrl_dim": 3,
                             "lr": 1e-3, "batch": 1024, "epochs": 1000},
}

# -----------------------------------------------------------------------------#
# main routine                                                                 #
# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser("Team‑PSRO runner (improved)")
    parser.add_argument("--dyn", default="single_integrator",
                        choices=list(DYN_TABLE.keys()),
                        help="dynamics model")
    parser.add_argument("--iters", type=int, default=80, help="PSRO iterations")
    parser.add_argument("--epochs", type=int, help="oracle training epochs")
    parser.add_argument("--batch", type=int, help="oracle RL batch size")
    parser.add_argument("--lr", type=float, help="oracle learning rate")
    parser.add_argument("--ego-start", nargs=2, type=float, metavar=("X", "Y"),
                        help="fixed start position (x y) for the ego team in [-1,1]")
    parser.add_argument("--opp-start", nargs=2, type=float, metavar=("X", "Y"),
                        help="fixed start position (x y) for the opponent team in [-1,1]")
    args = parser.parse_args()

    # pick defaults or CLI overrides
    dyn_cfg = DYN_TABLE[args.dyn]
    dyn_fn     = dyn_cfg["fn"]
    state_dim  = dyn_cfg["state_dim"]
    ctrl_dim   = dyn_cfg["ctrl_dim"]
    epochs     = args.epochs or dyn_cfg["epochs"]
    batch_size = args.batch  or dyn_cfg["batch"]
    lr         = args.lr     or dyn_cfg["lr"]

    # use fixed start positions if provided, otherwise fall back to defaults
    if args.ego_start:
        ego_start = list(args.ego_start)
    else:
        ego_start = [-1.0, -1.0]   # default fixed position for ego team

    if args.opp_start:
        opp_start = list(args.opp_start)
    else:
        opp_start = [1.0, 1.0]     # default fixed position for opponent team
        


    cfg = ConfigTeam(ego_start, opp_start)
    cfg.fsp_iteration = args.iters
    cfg.epochs = epochs
    cfg.batch_size = batch_size
    cfg.lr = lr

    # WANDB run
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{args.dyn}_run_{ts}"
    wandb.init(project=f"team-stl-psro_{args.dyn}", name=run_name,
               config={"iters": args.iters, "epochs": epochs,
                       "batch": batch_size, "lr": lr})
    print(f"EGO STARTING @ {ego_start}")
    print(f"OPP STARTING @ {opp_start}")
    driver = PSRODriver(cfg, dyn_fn, state_dim, ctrl_dim, team_size=2)

    # track robustness of newest BR pair
    for it in trange(args.iters, desc="Main PSRO loop"):
        exploit = driver.iterate()
        if hasattr(driver, "flush_payoff_queue"):
            driver.flush_payoff_queue()

        wandb.log({"iteration": it, "exploitability": exploit})

        # robustness of latest BR pair
        rob_br = driver._rollout_robustness(driver.pop_A[-1], driver.pop_B[-1])
        wandb.log({"robustness_br": rob_br}, commit=False)

        # adaptive GIF cadence: dense early, then every 20 iters
        if it < 20 or it % 20 == 0 or it == args.iters - 1:
            T_vis = cfg.T
            # ------------------------------------------------------------------
            # deterministic initial states for visualisation (B = 1)
            B_vis      = 1
            state_dim  = driver.env.state_dim
            total_sd   = driver.env.n_agents * state_dim
            init_vis   = torch.zeros(B_vis, total_sd, device=driver.cfg.device)

            # order: ego-0, ego-1, opp-0, opp-1
            init_vis[0, 0:2] = torch.tensor(ego_start)  # ego   agent 0
            init_vis[0, 2:4] = torch.tensor(ego_start)  # ego   agent 1
            init_vis[0, 4:6] = torch.tensor(opp_start)  # opp   agent 0
            init_vis[0, 6:8] = torch.tensor(opp_start)  # opp   agent 1
            # ------------------------------------------------------------------
            driver.env.reset(init_vis)                            # set the fixed start
            traj, _ = batched_rollout(driver.env,
                       driver.pop_A[-1], driver.pop_B[-1],
                       T=T_vis)
            sd = driver.env.state_dim
            pos_dim = 2
            ego = [traj[:, :, i*sd:(i*sd)+pos_dim] for i in range(2)]
            opp = [traj[:, :, (2+i)*sd:(2+i)*sd+pos_dim] for i in range(2)]
            gif_path = os.path.join("artifact/figs/results/main",
                                    f"{args.dyn}_psro_{it}.gif")
            animate(ego, opp, driver.cfg.get_obstacles(),
                    driver.cfg.get_obstacles()[2], gif_path, interval_ms=250)
            wandb.save(gif_path)

    # final robustness & GIF
    T_vis = cfg.T
    # ------------------------------------------------------------------
    # deterministic initial states for visualisation (B = 1)
    B_vis      = 1
    state_dim  = driver.env.state_dim
    total_sd   = driver.env.n_agents * state_dim
    init_vis   = torch.zeros(B_vis, total_sd, device=driver.cfg.device)

    # order: ego-0, ego-1, opp-0, opp-1
    init_vis[0, 0:2] = torch.tensor(ego_start)  # ego   agent 0
    init_vis[0, 2:4] = torch.tensor(ego_start)  # ego   agent 1
    init_vis[0, 4:6] = torch.tensor(opp_start)  # opp   agent 0
    init_vis[0, 6:8] = torch.tensor(opp_start)  # opp   agent 1
    # ------------------------------------------------------------------
    driver.env.reset(init_vis)                            # set the fixed start
    traj, _ = batched_rollout(driver.env,
                driver.pop_A[-1], driver.pop_B[-1],
                T=T_vis)
    sd = driver.env.state_dim
    pos_dim = 2
    ego = [traj[:, :, i*sd:(i*sd)+pos_dim] for i in range(2)]
    opp = [traj[:, :, (2+i)*sd:(2+i)*sd+pos_dim] for i in range(2)]

    rob_ego = driver.stl.compute_robustness_ego(ego, opp)
    rob_opp = driver.stl.compute_robustness_opp(ego, opp)
    wandb.log({"robustness_ego_final": rob_ego.item(),
               "robustness_opp_final": rob_opp.item()})

    save_dir = "artifact/figs/results/main"
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, f"{args.dyn}_psro_final.gif")
    animate(ego, opp, driver.cfg.get_obstacles(),
            driver.cfg.get_obstacles()[2], gif_path, interval_ms=250)
    wandb.save(gif_path)
    wandb.finish()


if __name__ == "__main__":
    main()
