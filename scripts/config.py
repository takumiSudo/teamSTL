import torch
import os

# obstacles
obs_1 = torch.tensor([0.0, 0.9, -1.0, -0.5]).float()     # red box in bottom right corner
obs_2 = torch.tensor([.2, 0.7, 0.8, 1.2]).float()        # green box in top right corner
obs_3 = torch.tensor([0.0, 0.0, 0.4]).float()            # blue circle in the center
obs_4 = torch.tensor([-1.0, -0.7, -0.2, 0.5]).float()    # orange box on the left
goal = torch.tensor([0.98, 1.02, 0.98, 1.02]).float()

# parameters and dir
T = 5
total_time_step = 50
u_max = torch.as_tensor(0.8).float()       # u max
model_dir = 'models/car/'
data_dir = 'data/gradient_exp/car/'
exp_fig_dir = 'figs/exp/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(exp_fig_dir, exist_ok=True)

# initial agents position choice for 2d system:
opp_choice1 = [-0.5, -0.5, 0.5, 0.5]
opp_choice2 = [0., 0., 0.5, 0.5]
opp_choice3 = [-0.25, -0.25, 0.5, 0.5]
opp_choice4 = [-0.5, -1., 0.5, 1.]
opp_choice5 = [-0.5, -0.9, 0.5, 1.]

ego_choice1 = [-1., -1., 0.5, 1.]
ego_choice2 = [-0.5, -1.0, 0.5, 1.]
ego_choice3 = [-1., -0.5, 0.5, 1.]
ego_choice4 = [-0.75, -0.75, 0.5, 1.]
ego_choice5 = [-0.5, -0.75, 0.5, 1.]

ego_choice = ego_choice2
opp_choice = opp_choice2

# ref trajectory
x_ego_init, y_ego_init = ego_choice[0], ego_choice[1]
x_ego_goal, y_ego_goal = ego_choice[2], ego_choice[3]
x_ego = torch.arange(x_ego_init, x_ego_goal, (x_ego_goal-x_ego_init)/total_time_step).reshape(-1, 1)
y_ego = torch.arange(y_ego_init, y_ego_goal, (y_ego_goal-y_ego_init)/total_time_step).reshape(-1, 1)
ref_X_ego = torch.cat((x_ego, y_ego), dim=1)

x_opp_init, y_opp_init = opp_choice[0], opp_choice[1]
x_opp_goal, y_opp_goal = opp_choice[2], opp_choice[3]
x_opp = torch.arange(x_opp_init, x_opp_goal, (x_opp_goal-x_opp_init)/total_time_step).reshape(-1, 1)
y_opp = torch.arange(y_opp_init, y_opp_goal, (y_opp_goal-y_opp_init)/total_time_step).reshape(-1, 1)
ref_X_opp = torch.cat((x_opp, y_opp), dim=1)

# network
hiddent_dim=32
pred_len = 1

# self-play train
fsp_iteration = 50