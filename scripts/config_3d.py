import torch
import os
import math

# obstacles
obs = torch.tensor([0.0, 1., 0.0, 1., 0.0, 2.0]).float()     # red box in bottom right corner
goal = torch.tensor([1.5, 2.5, 1.5, 2.5, 0.0, 0.5]).float()

zone1 = torch.tensor([-1, 1]).float()
zone2 = torch.tensor([1, 3]).float()
altitude1 = torch.tensor([1, 5]).float()
altitude2 = torch.tensor([0, 3]).float()

# parameters and dir
T = 5
total_time_step = 50
model_dir = 'models/drone/'
data_dir = 'data/gradient_exp/drone/'
exp_fig_dir = 'figs/exp/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(exp_fig_dir, exist_ok=True)

# initial agents position choice for drone system:
opp_choice1 = [0., 0.5, 1.3, goal[0], goal[2], goal[5]]
opp_choice2 = [0., 0., 1.1, goal[0], goal[2], goal[5]]
opp_choice3 = [-0.25, -0.25, 0.8, goal[0], goal[2], goal[5]]
opp_choice4 = [-0.5, -1., 0.8, goal[0], goal[2], goal[5]]
opp_choice5 = [0.5, -0.9, 1.4, goal[0], goal[2], goal[5]]

ego_choice1 = [-1., -1., 1.4, goal[0], goal[2], goal[5]]
ego_choice2 = [-0.5, -1.0, 1.1, goal[0], goal[2], goal[5]]
ego_choice3 = [-1., -0.5, 1.5, goal[0], goal[2], goal[5]]
ego_choice4 = [0.5, -0.75, 0.2, goal[0], goal[2], goal[5]]
ego_choice5 = [0.0, -0.75, 1.2, goal[0], goal[2], goal[5]]

ego_choice = ego_choice1
opp_choice = opp_choice1

# ref trajectory
x_ego_init, y_ego_init, z_ego_init = ego_choice[0], ego_choice[1], ego_choice[2]
x_ego_goal, y_ego_goal, z_ego_goal = ego_choice[3], ego_choice[4], ego_choice[5]
x_ego = torch.arange(x_ego_init, x_ego_goal, (x_ego_goal-x_ego_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
y_ego = torch.arange(y_ego_init, y_ego_goal, (y_ego_goal-y_ego_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
z_ego = torch.arange(z_ego_init, z_ego_goal, (z_ego_goal-z_ego_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
ref_X_ego = torch.cat((x_ego, y_ego, z_ego), dim=1)

x_opp_init, y_opp_init, z_opp_init = opp_choice[0], opp_choice[1], opp_choice[2]
x_opp_goal, y_opp_goal, z_opp_goal = opp_choice[3], opp_choice[4], opp_choice[5]
x_opp = torch.arange(x_opp_init, x_opp_goal, (x_opp_goal-x_opp_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
y_opp = torch.arange(y_opp_init, y_opp_goal, (y_opp_goal-y_opp_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
z_opp = torch.arange(z_opp_init, z_opp_goal, (z_opp_goal-z_opp_init)/total_time_step).reshape(-1, 1)[:total_time_step, :]
ref_X_opp = torch.cat((x_opp, y_opp, z_opp), dim=1)

# network
hiddent_dim=32
pred_len = 1

# self-play train
fsp_iteration = 100