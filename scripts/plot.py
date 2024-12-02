import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

sns.set_context("poster")
sns.set_style("ticks")

font_size = 20
# plot the mean & std for exploitability

def plot_exploitability_avg(dynamic, exp_list_grad, exp_list_rl, fig_dir):

    data = np.load(exp_list_grad[0]+'.npy')
    for i in range(1, len(exp_list_grad), 1):
        new_data_block = np.load(exp_list_grad[i]+'.npy')
        data = np.vstack([data, new_data_block])
    data = np.transpose(data)

    if exp_list_rl is not None:
        rl_data = np.load(exp_list_rl[0]+'.npy')
        for i in range(1, len(exp_list_rl), 1):
            new_data_block = np.load(exp_list_rl[i]+'.npy')
            rl_data = np.vstack([rl_data, new_data_block])
        rl_data = np.transpose(rl_data)

    # Plot the mean and std
    plt.figure(figsize=(8, 6))

    # convert data to a pandas dataframe that seaborn can use with steps as one of the columns
    data_w_steps = np.hstack((np.arange(data.shape[0]).reshape(-1, 1), data))
    data_df = pd.DataFrame(data_w_steps, columns=["FSP Iterations", "0", "1", "2", "3", "4"])
    data_df = data_df.melt(value_vars=["0", "1", "2", "3", "4"], id_vars=["FSP Iterations"], value_name="Exploitability")
    sns.lineplot(data=data_df, x="FSP Iterations", y="Exploitability", label='Gradient-based method', 
                 color='darkslateblue', linestyle='-', linewidth=2, errorbar='sd')

    if exp_list_rl is not None:
        rl_data_w_steps = np.hstack((np.arange(rl_data.shape[0]).reshape(-1, 1), rl_data))
        rl_data_df = pd.DataFrame(rl_data_w_steps, columns=["FSP Iterations", "0", "1", "2", "3", "4"])
        rl_data_df = rl_data_df.melt(value_vars=["0", "1", "2", "3", "4"], id_vars=["FSP Iterations"], value_name="Exploitability")
        sns.lineplot(data=rl_data_df, x="FSP Iterations", y="Exploitability", label='Reinforcement learning', color='deeppink', linestyle='-', linewidth=2, errorbar='sd')

    if dynamic == 'drone':
        title_name = 'Autonomous Drones'
        fig_save_name = 'mean_exp_drone.pdf'
    else:
        title_name = 'Ackermann Steering Vehicles'
        fig_save_name = 'mean_exp_car.pdf'

    # Add labels and title
     
    plt.xlabel('FSP iterations', fontsize=font_size)
    plt.ylabel('Exploitability', fontsize=font_size)
    plt.title(f'Exploitability for {title_name}', fontsize=font_size)

    # Show the legend
    plt.legend(prop={'size': font_size}, loc='upper left')
    plt.savefig(fig_dir+fig_save_name, format="pdf", bbox_inches="tight")
    plt.close()

dynamic = 'car'
name = f'data/gradient_exp/car/exploitability_kinematic_'
exp_name_list_grad = [name+'1', name+'2', name+'3', name+'4', name+'5']
name = f'data/rl_exp/car/exploitability'
exp_name_list_rl = [name+'1', name+'2', name+'3', name+'4', name+'5']

name = 'data/gradient_exp/drone/exploitability_3d_rotor_'
exp_name_list_3d_grad = [name+'1', name+'2', name+'3', name+'4', name+'5']
name = f'data/rl_exp/{dynamic}/exploitability'
exp_name_list_3d_rl = [name+'1', name+'2', name+'3', name+'4', name+'5']

if dynamic == 'drone':
    exp_grad = exp_name_list_3d_grad
    exp_rl = exp_name_list_3d_rl
else:
    exp_grad = exp_name_list_grad
    exp_rl = exp_name_list_rl

plot_exploitability_avg(dynamic, exp_grad, exp_rl, 'figs/exp/')