import torch
import os
import random

# obstacles
obstacles = [
    torch.tensor([0.0, 0.9, -1.0, -0.5]).float(),     # red box in bottom right corner
    torch.tensor([.2, 0.7, 0.8, 1.2]).float(),        # green box in top right corner
    torch.tensor([0.0, 0.0, 0.4]).float(),            # blue circle in the center
    torch.tensor([-1.0, -0.7, -0.2, 0.5]).float(),    # orange box on the left
    torch.tensor([0.98, 1.02, 0.98, 1.02]).float()    # goal
]

class ConfigTeam:
    def __init__(self, ego_choices, opp_choices, total_time_step=50, T=5, u_max=0.8, 
                 model_dir='models/team/', data_dir='data/team/', exp_fig_dir='figs/team/'):
        self.ego_choices = ego_choices
        self.opp_choices = opp_choices
        self.total_time_step = total_time_step
        self.T = T
        self.u_max = torch.as_tensor(u_max).float()
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.exp_fig_dir = exp_fig_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.exp_fig_dir, exist_ok=True)
        
        self.hidden_dim = 32
        self.pred_len = 1
        self.fsp_iteration = 50
        
    def compute_ref_trajs(self):
        ref_X_ego = []
        for choice in self.ego_choices:
            x_init, y_init = choice[0], choice[1]
            x_goal, y_goal = choice[2], choice[3]
            x = torch.arange(x_init, x_goal, (x_goal - x_init) / self.total_time_step).reshape(-1, 1)
            y = torch.arange(y_init, y_goal, (y_goal - y_init) / self.total_time_step).reshape(-1, 1)
            ref_X_ego.append(torch.cat((x, y), dim=1))
        
        ref_X_opp = []
        for choice in self.opp_choices:
            x_init, y_init = choice[0], choice[1]
            x_goal, y_goal = choice[2], choice[3]
            x = torch.arange(x_init, x_goal, (x_goal - x_init) / self.total_time_step).reshape(-1, 1)
            y = torch.arange(y_init, y_goal, (y_goal - y_init) / self.total_time_step).reshape(-1, 1)
            ref_X_opp.append(torch.cat((x, y), dim=1))
        
        return ref_X_ego, ref_X_opp
    
    def get_obstacles(self):
        return obstacles


def generate_start_end_positions():
    x1 = random.uniform(-1, 1)
    y1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y2 = random.uniform(-1, 1)
    return [x1, y1, x2, y2]

if __name__ == "__main__":
    ego_choices = [generate_start_end_positions() for _ in range(1)]
    opp_choices = [generate_start_end_positions() for _ in range(2)]
    cfg = ConfigTeam(ego_choices, opp_choices)
    
    ref_X_ego, ref_X_opp = cfg.compute_ref_trajs()
    print("Ego Reference Trajectories:")
    for traj in ref_X_ego:
        print(traj)
    
    print("\nOpponent Reference Trajectories:")
    for traj in ref_X_opp:
        print(traj)
    
    print("\nObstacles:")
    for obs in cfg.get_obstacles():
        print(obs)