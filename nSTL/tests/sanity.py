# ------------------------------------------------------------
#  quick_sanity.py
#
#  Unit-test & visualise STLFormulaReachAvoidTeam for
#  1 v 2, 2 v 1 and 2 v 2 scenarios.
#
#  Run:  python quick_sanity.py
# ------------------------------------------------------------
import torch, random, os
import matplotlib.pyplot as plt
from matplotlib import animation

from nSTL.config.team_config import ConfigTeam, generate_start_end_positions
from nSTL.robust.team_stl_helper import STLFormulaReachAvoidTeam

# -------- parameters ----------------------------------------
TOTAL_T    = 50
SAFE_D     = torch.tensor(0.25)
SAVE_DIR   = "figs/team/tests/"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- helper: linear interp traj ------------------------
def line_traj(start_xy, goal_xy, T=TOTAL_T):
    xs = torch.linspace(start_xy[0], goal_xy[0], T).reshape(-1,1)
    ys = torch.linspace(start_xy[1], goal_xy[1], T).reshape(-1,1)
    return torch.cat((xs,ys), dim=1)

# -------- helper: animate -----------------------------------
def animate_scene(ego_trajs, opp_trajs, obstacles, circle, fname):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.set_aspect("equal"); ax.grid()

    # plot obstacles once
    obs1, obs2 = obstacles[:2]
    x1_lo, x1_hi, y1_lo, y1_hi = obs1[0].item(), obs1[1].item(), obs1[2].item(), obs1[3].item()
    x2_lo, x2_hi, y2_lo, y2_hi = obs2[0].item(), obs2[1].item(), obs2[2].item(), obs2[3].item()

    ax.plot([x1_lo, x1_lo, x1_hi, x1_hi, x1_lo],
            [y1_lo, y1_hi, y1_hi, y1_lo, y1_lo], "r")

    ax.plot([x2_lo, x2_lo, x2_hi, x2_hi, x2_lo],
            [y2_lo, y2_hi, y2_hi, y2_lo, y2_lo], "g")

    cx, cy, rad = circle[0].item(), circle[1].item(), circle[2].item()
    ang = torch.linspace(0,2*torch.pi,100)
    ax.plot(cx + rad*torch.cos(ang),
            cy + rad*torch.sin(ang),"b")

    ego_lines, = ax.plot([],[],'-o',c='orange',label='ego')
    opp_lines, = ax.plot([],[],'-o',c='purple',label='opp')
    ax.legend()

    def init():
        ego_lines.set_data([],[]); opp_lines.set_data([],[])
        return ego_lines, opp_lines
    def animate(t):
        # concat all egos / opps for scatter
        ego_xy = torch.vstack([traj[t] for traj in ego_trajs])
        opp_xy = torch.vstack([traj[t] for traj in opp_trajs])
        ego_lines.set_data(ego_xy[:,0], ego_xy[:,1])
        opp_lines.set_data(opp_xy[:,0], opp_xy[:,1])
        return ego_lines, opp_lines
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=TOTAL_T, interval=80, blit=True)
    anim.save(os.path.join(SAVE_DIR,fname), writer="pillow")
    plt.close()

# -------- build common obstacles ----------------------------
cfg_tmp    = ConfigTeam([],[])
OBS        = cfg_tmp.get_obstacles()
CIRCLE     = OBS[2]; GOAL = OBS[4]

# -------- scenarios -----------------------------------------
scenarios = {
    "1v2": (1,2),
    "2v1": (2,1),
    "2v2": (2,2)
}

for tag,(m,n) in scenarios.items():
    ego_choices = [generate_start_end_positions() for _ in range(m)]
    opp_choices = [generate_start_end_positions() for _ in range(n)]
    ego_trajs   = [line_traj(c[:2],c[2:], TOTAL_T) for c in ego_choices]
    opp_trajs   = [line_traj(c[:2],c[2:], TOTAL_T) for c in opp_choices]

    stl = STLFormulaReachAvoidTeam(
            obs_boxes=[OBS[0], OBS[1]],
            circle_obs=CIRCLE,
            goal_box = GOAL,
            T        = cfg_tmp.T,
            safe_d   = SAFE_D)

    rob_ego = stl.compute_robustness_ego(ego_trajs, opp_trajs)
    rob_opp = stl.compute_robustness_opp(ego_trajs, opp_trajs)

    print(f"\n=== {tag} ===")
    print(f"Ego robustness: {rob_ego.item(): .3f}")
    print(f"Opp robustness: {rob_opp.item(): .3f}")

    animate_scene(ego_trajs, opp_trajs, OBS, CIRCLE,
                  fname=f"{tag}.gif")
    print(f"animation saved to {SAVE_DIR}{tag}.gif")

if __name__ == "__main__":
    pass  