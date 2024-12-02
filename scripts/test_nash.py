import torch
import numpy as np
import math
from utils.modules_helper import LSTM
from utils.dynamic_helper import single_integrator, double_integrator, double_integrator_3d, rollout_batch_two_agents
from utils.dynamic_helper import kinematic_model, quadrotor
from utils.stl_helper import STLFormulaReachAvoidTwoAgents, STLFormulaReachAvoidTwoAgents3D
import argparse
import copy
np.random.seed(0)

def normalize_sample_prob(lst):
    return [float(i)/sum(lst) for i in lst]

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

    # models path
    model_filename_ego = model_dir+'policy_ego_'+args.dynamic
    model_filename_opp = model_dir+'policy_opp_'+args.dynamic

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
        
        test_batch_size = 150

        # choose ego and opp policy types that you want to test
        test_ego_method = 'nash'
        # test_ego_method = 'br'
        test_opp_method = 'seen'
        test_opp_method = 'unseen'

        # ego and opponent play against each other
        for test_index in range(test_batch_size):
            if test_ego_method == 'nash':
                new_p_sample = normalize_sample_prob(p_sample[:args.nash_iter])
                policy_ego = np.random.choice(ego_model_list[:args.nash_iter], p=new_p_sample)
            elif test_ego_method == 'br':
                policy_ego = ego_model_list[args.iter]
            else:
                raise ValueError(f"{test_ego_method} is not defined")

            if test_opp_method == 'seen':
                new_p_sample = normalize_sample_prob(p_sample[:args.iter])
                policy_opp = np.random.choice(opp_model_list[:args.iter], p=new_p_sample)
            elif test_opp_method == 'unseen':
                new_p_sample = normalize_sample_prob(p_sample[args.nash_iter:])
                policy_opp = np.random.choice(opp_model_list[args.nash_iter:], p=new_p_sample)
                if args.dynamic == 'kinematic_model':
                    unseen_index = test_index % 4 + 1
                    checkpoint_opp = torch.load(f'models/car/kinematic_opp_unseen{unseen_index}.pth', weights_only=True)
                    policy_br_ego.load_state_dict(checkpoint_opp['model_state_dict'])
                    policy_opp = policy_br_ego
            else:
                raise ValueError(f"{test_opp_method} is not defined")

            current_traj_idx, _, _ = rollout_batch_two_agents(
                                                        args=args,
                                                        init_state_ego=ref_X_ego[0:1, :],
                                                        init_state_opp=ref_X_opp[0:1, :],
                                                        policy_ego=policy_ego,
                                                        policy_opp=policy_opp,
                                                        opp_noise=True
                                                        )
            if test_index == 0:
                current_traj = current_traj_idx
            else:
                current_traj = torch.cat((current_traj, current_traj_idx), dim=0)

        current_traj_ego = current_traj[..., :state_dim]
        current_traj_opp = current_traj[..., state_dim:]
        
        robustness = stl_class.compute_robustness_ego(current_traj_ego[..., :traj_dim], current_traj_opp[..., :traj_dim]).squeeze()
        print(f"Ego use {test_ego_method}, opp use {test_opp_method}")
        print(f"Test STL satisfaction rate {torch.where(robustness>=0.)[0].shape[0] / robustness.shape[0]}")
        print(f"Test STL robustness std & mean {torch.std_mean(robustness)}.")

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
        seen_num, nash_iter_num = 35, 49
        from config import *
    elif args.dynamic == 'dt3d':
        dynamic_ego, dynamic_opp = double_integrator_3d, double_integrator_3d
        state_dim, control_dim, traj_dim, action_max = 6, 3, 3, 0.8
        from config_3d import *
    elif args.dynamic == 'rotor':
        dynamic_ego, dynamic_opp = quadrotor, quadrotor
        state_dim, control_dim, traj_dim, action_max = 6, 3, 3, torch.tensor([30*(math.pi/180), 30*(math.pi/180), 0.15]).float()
        seen_num, nash_iter_num = 25, 75
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
    parser.add_argument('--iter', default=seen_num, help='Seen policies num')
    parser.add_argument('--nash_iter', default=nash_iter_num, help='Policy after nash_iter will be considered as unseen')    

    args = parser.parse_args()
    
    main(args)