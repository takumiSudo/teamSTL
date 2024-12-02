import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, math
sys.path.append('./')
from utils.modules_helper import LSTM
from utils.dynamic_helper import single_integrator, double_integrator, double_integrator_3d, rollout_batch_two_agents
from utils.dynamic_helper import kinematic_model, quadrotor
from utils.stl_helper import STLFormulaReachAvoidTwoAgents, STLFormulaReachAvoidTwoAgents3D
from utils.train_helper import Trainer
import argparse
import copy
np.random.seed(0)

def plot_exploitability(exp_list, fig_dir):
    list_len = exp_list.shape[0]
    plt.plot(np.arange(0, list_len, 1), exp_list, marker = 'o')
    plt.savefig(fig_dir+'exp.png')
    plt.close()

def main(args):

    # stl specification
    if args.dynamic == 'dt3d' or args.dynamic == 'rotor':
        stl_class = STLFormulaReachAvoidTwoAgents3D(obs, goal, T, zone1, zone2, altitude1, altitude2)
    else:
        stl_class = STLFormulaReachAvoidTwoAgents(obs_1, obs_2, obs_3, obs_4, goal, T)

    # current best response policy for ego and opponent
    policy_br_ego = LSTM(hidden_dim=hiddent_dim, pred_length=pred_len, state_dim=2*args.state_dim, action_dim=args.control_dim, 
                        stochastic_policy=False, batch_first=args.batch_first, action_max=args.action_max)
    policy_br_opp = LSTM(hidden_dim=hiddent_dim, pred_length=pred_len, state_dim=2*args.state_dim, action_dim=args.control_dim,
                        stochastic_policy=False, batch_first=args.batch_first, action_max=args.action_max)

    # initial (random) policies for ego and opponent
    policy_init_ego = LSTM(hidden_dim=hiddent_dim, pred_length=pred_len, state_dim=2*args.state_dim, action_dim=args.control_dim, 
                        stochastic_policy=False, batch_first=args.batch_first, action_max=args.action_max)
    policy_init_opp = LSTM(hidden_dim=hiddent_dim, pred_length=pred_len, state_dim=2*args.state_dim, action_dim=args.control_dim, 
                        stochastic_policy=False, batch_first=args.batch_first, action_max=args.action_max)
    
    # policy sample distribution, to be updated as the FSP iterates
    policy_avg_sample_dist = [1]

    # models path
    model_filename_ego = model_dir+'policy_ego_'+args.dynamic
    model_filename_opp = model_dir+'policy_opp_'+args.dynamic

    # build the trainer (without using supervised learning for average policy)
    trainer = Trainer(args=args, stl_class=stl_class, policy_avg_ego=[policy_init_ego], policy_avg_opp=[policy_init_opp],
                          p_sample_dist=policy_avg_sample_dist, policy_br_ego=policy_br_ego, policy_br_opp=policy_br_opp)

    # train
    if not args.only_test:
        
        exploitability_list = np.zeros(fsp_iteration) # store explotability over FSP iterations

        # fictitious self play loop
        for fsp_iteration_index in range(fsp_iteration):
            for agent_name in ['ego', 'opp']:
                exploitability, p_sample = trainer.train(args, iteration=fsp_iteration_index, save_path_ego=model_filename_ego, 
                                                         save_path_opp=model_filename_opp, whose_turn=agent_name)
                if agent_name == 'opp':
                    exploitability_list[fsp_iteration_index] = exploitability
            np.save(data_dir+'exploitability.npy', exploitability_list)
            plot_exploitability(exploitability_list, exp_fig_dir)

    # test
    with torch.no_grad():

        # load models and update sample distribution
        ego_model_list, opp_model_list = [], []
        p_sample = [1]
        for fsp_iteration_index in range(fsp_iteration):
            policy_br_ego_iter = copy.deepcopy(policy_br_ego)
            policy_br_opp_iter = copy.deepcopy(policy_br_opp)
            checkpoint_ego = torch.load(model_filename_ego+f'_best_iter{fsp_iteration_index}.pth', weights_only=True)
            checkpoint_opp = torch.load(model_filename_opp+f'_best_iter{fsp_iteration_index}.pth', weights_only=True)
            policy_br_ego_iter.load_state_dict(checkpoint_ego['model_state_dict'])
            policy_br_opp_iter.load_state_dict(checkpoint_opp['model_state_dict'])
            ego_model_list.append(policy_br_ego_iter)
            opp_model_list.append(policy_br_opp_iter)

            if fsp_iteration_index >= 1:
                current_time = fsp_iteration_index + 1
                avg_policy_prob = (current_time - 1) / (current_time + 1)
                latest_policy_prob = 2 / (current_time + 1)
                p_sample.append(1 / avg_policy_prob * latest_policy_prob)
                p_sample = [float(i)/sum(p_sample) for i in p_sample]
        
        test_batch_size = 100

        # ego and opponent play against each other
        for test_index in range(test_batch_size):
            policy_ego = np.random.choice(ego_model_list, p=p_sample)
            policy_opp = np.random.choice(opp_model_list, p=p_sample)

            current_traj_idx, _, _ = rollout_batch_two_agents(
                                                        args=args,
                                                        init_state_ego=ref_X_ego[0:1, :],
                                                        init_state_opp=ref_X_opp[0:1, :],
                                                        policy_ego=policy_ego,
                                                        policy_opp=policy_opp,
                                                        )
            if test_index == 0:
                current_traj = current_traj_idx
            else:
                current_traj = torch.cat((current_traj, current_traj_idx), dim=0)

        current_traj_ego = current_traj[..., :state_dim]
        current_traj_opp = current_traj[..., state_dim:]
        
        robustness = stl_class.compute_robustness_ego(current_traj_ego[..., :traj_dim], current_traj_opp[..., :traj_dim]).squeeze()
        print(f"Test STL satisfaction num {torch.where(robustness>=0.)[0].shape[0]} out of {robustness.shape[0]} trajectories.")
        print(f"Test STL robustness mean {torch.mean(robustness).item()}.")
        test_index_list = [1]
        for test_index in test_index_list:
            stl_class.animate(current_traj_ego[test_index, ...], current_traj_opp[test_index, ...], name=args.fig_name+str(test_index))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='stl_seq_policy')
    parser.add_argument('--only_test', action=argparse.BooleanOptionalAction, help='Dont train policy')
    parser.add_argument('--fig_name', type=str, default='test_multi_agent', help='Figure name')
    parser.add_argument('--dynamic', type=str, default='st', help='Single integrator or double')
    parser.add_argument('--total_time_step', type=int, default=50, help='Agents simulation steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Best response training learn rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Data collecting batch size')
    parser.add_argument('--train_epoch', type=int, default=201, help='Training epochs num')
    parser.add_argument('--robust_margin', type=float, default=0.1, help='STL robustness margin')
    parser.add_argument('--ref_loss_weight', type=float, default=0.2, help='Loss weight of following the ref traj')
    parser.add_argument('--batch_first', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_data_gen_fn', default=rollout_batch_two_agents, help='Data generation function')

    args = parser.parse_args()


    # dynamic
    if args.dynamic == 'st':
        dynamic_ego, dynamic_opp = single_integrator, single_integrator
        state_dim, control_dim, traj_dim, action_max = 2, 2, 2, 0.8
    elif args.dynamic == 'dt':
        dynamic_ego, dynamic_opp = double_integrator, double_integrator
        state_dim, control_dim, traj_dim, action_max = 4, 2, 2, 0.8
        from config import *
    elif args.dynamic == 'kinematic_model':
        dynamic_ego, dynamic_opp = kinematic_model, kinematic_model
        state_dim, control_dim, traj_dim, action_max = 5, 2, 2, torch.tensor([0.3, 7.5]).float()
        from config import *
    elif args.dynamic == 'dt3d':
        dynamic_ego, dynamic_opp = double_integrator_3d, double_integrator_3d
        state_dim, control_dim, traj_dim, action_max = 6, 3, 3, 0.8
        from config_3d import *
    elif args.dynamic == 'rotor':
        dynamic_ego, dynamic_opp = quadrotor, quadrotor
        state_dim, control_dim, traj_dim, action_max = 6, 3, 3, torch.tensor([30*(math.pi/180), 30*(math.pi/180), 0.15]).float()
        from config_3d import *
    else:
        raise ValueError(f"{args.dynamic} not defined")

    # update args
    parser.add_argument('--state_dim', type=int, default=state_dim)
    parser.add_argument('--control_dim', type=int, default=control_dim)
    parser.add_argument('--traj_dim', type=int, default=traj_dim)
    parser.add_argument('--action_max', default=action_max)
    parser.add_argument('--dynamic_ego', default=dynamic_ego)
    parser.add_argument('--dynamic_opp', default=dynamic_opp)
    parser.add_argument('--ref_ego', type=list, default=ref_X_ego, help='Ego task-unaware reference trajectory')
    parser.add_argument('--ref_opp', type=list, default=ref_X_opp, help='Opponent task-unaware reference trajectory')
    
    args = parser.parse_args()
    
    main(args)