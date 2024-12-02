import torch
import numpy as np
import random
import copy
random.seed(0)
np.random.seed(seed=0)

# single agent stochastic policy loss function
def stochastic_loss_fn(stl_class, current_traj, log_prob_seq, control_seq, ref_traj,
                       ref_loss_weight=1, ctrl_loss_weight=0, margin=0.1, u_max=0.8):

    stl_roustness = stl_class.compute_robustness(current_traj).squeeze()
    stl_reward = - torch.relu(-stl_roustness + margin)
    stl_loss = torch.mean(- stl_reward * torch.sum(log_prob_seq, (2,1)))
    ref_reward = - torch.square((current_traj - ref_traj))[:, :-1, :]
    ref_loss = torch.mean(- ref_reward * log_prob_seq)
    control_loss = torch.norm(torch.relu(control_seq - u_max))
    loss = stl_loss + ref_loss_weight*ref_loss + ctrl_loss_weight*control_loss

    return loss, stl_loss, ref_loss, control_loss, stl_roustness

# single agent deterministic policy loss function
def deterministic_loss_fn(stl_class, current_traj, log_prob_seq, control_seq, ref_traj, 
                          ref_loss_weight=1, ctrl_loss_weight=0, margin=0.1, u_max=0.8):

    stl_roustness = stl_class.compute_robustness(current_traj).squeeze()
    stl_loss = torch.relu(-stl_roustness + margin)
    ref_loss = torch.nn.MSELoss()(current_traj, ref_traj)
    control_loss = torch.norm(torch.relu(control_seq - u_max))
    loss = stl_loss + ref_loss_weight*ref_loss + ctrl_loss_weight*control_loss

    return loss, stl_loss, ref_loss, control_loss, stl_roustness

# multi agent stochastic policy loss function for ego
def stochastic_loss_fn_multi_agent_ego(stl_class, ego_traj, opp_traj, log_prob_seq_ego, log_prob_seq_opp, 
                                   control_seq, ref_traj_ego, ref_traj_opp,
                                   ref_loss_weight=1, ctrl_loss_weight=0, margin=0.1, u_max=0.8):

    stl_roustness = stl_class.compute_robustness(ego_traj, opp_traj).squeeze()
    stl_reward = - torch.relu(-stl_roustness + margin)
    stl_loss = torch.mean(- stl_reward * torch.sum(log_prob_seq_ego, (2,1)))
    ref_reward = - torch.square((ego_traj - ref_traj_ego))[:, :-1, :]
    ref_loss = torch.mean(- ref_reward * log_prob_seq_ego)
    control_loss = torch.norm(torch.relu(control_seq - u_max))
    loss = stl_loss + ref_loss_weight*ref_loss + ctrl_loss_weight*control_loss

    return loss, stl_loss, ref_loss, control_loss, stl_roustness

# multi agent deterministic policy loss function for ego
def deterministic_loss_fn_multi_agent_ego(args, stl_class, ego_traj, opp_traj):
    
    ref_traj_ego = args.ref_ego.repeat(ego_traj.shape[0], 1, 1)
    stl_roustness = stl_class.compute_robustness_ego(ego_traj, opp_traj).squeeze()
    stl_loss = torch.relu(-stl_roustness + args.robust_margin)
    ref_loss = torch.nn.MSELoss()(ego_traj, ref_traj_ego)
    loss = stl_loss + args.ref_loss_weight*ref_loss

    return loss, stl_loss, ref_loss, None, stl_roustness

# multi agent stochastic policy loss function for opponent
def stochastic_loss_fn_multi_agent_opp(stl_class, ego_traj, opp_traj, log_prob_seq_ego, log_prob_seq_opp, 
                                   control_seq, ref_traj_ego, ref_traj_opp,
                                   ref_loss_weight=1, ctrl_loss_weight=0, margin=0.1, u_max=0.8):

    stl_roustness = stl_class.compute_robustness(ego_traj, opp_traj).squeeze()
    stl_reward = - torch.relu(-stl_roustness + margin)
    stl_loss = - torch.mean(- stl_reward * torch.sum(log_prob_seq_opp, (2,1)))
    ref_reward = - torch.square((opp_traj - ref_traj_opp))[:, :-1, :]
    ref_loss = torch.mean(- ref_reward * log_prob_seq_opp)
    control_loss = torch.norm(torch.relu(control_seq - u_max))
    loss = stl_loss + ref_loss_weight*ref_loss + ctrl_loss_weight*control_loss

    return loss, stl_loss, ref_loss, control_loss, stl_roustness

# multi agent deterministic policy loss function for opponent
def deterministic_loss_fn_multi_agent_opp(args, stl_class, ego_traj, opp_traj):

    ref_traj_opp = args.ref_opp.repeat(opp_traj.shape[0], 1, 1)
    stl_roustness = stl_class.compute_robustness_opp(ego_traj, opp_traj).squeeze()
    stl_loss = - torch.relu(-stl_roustness + args.robust_margin)
    ref_loss = torch.nn.MSELoss()(opp_traj, ref_traj_opp)
    loss = stl_loss + args.ref_loss_weight*ref_loss

    return loss, stl_loss, ref_loss, None, stl_roustness


# The trainer we are using
class Trainer:

    def __init__(self, args, stl_class, policy_avg_ego=None, policy_avg_opp=None, p_sample_dist=None,
                 policy_br_ego=None, policy_br_opp=None):
        self.args = args
        self.stl_class = stl_class
        self.ref_X_ego = args.ref_ego
        self.ref_X_opp = args.ref_opp
        self.learn_rate = args.lr
        self.policy_avg_ego_set = policy_avg_ego
        self.policy_avg_opp_set = policy_avg_opp
        self.avg_policy_sample_dist = p_sample_dist
        self.policy_br_ego = policy_br_ego
        self.policy_br_opp = policy_br_opp

    def train(self, args, save_path_ego=None, save_path_opp=None, whose_turn="ego", iteration=0):        
        
        if iteration == 0:
            policy_br_ego = self.policy_br_ego
            policy_br_opp = self.policy_br_opp
        else:
            # load the best response model from last iteration for warmstarting
            policy_br_ego = copy.deepcopy(self.policy_br_ego)
            policy_br_opp = copy.deepcopy(self.policy_br_opp)
            checkpoint_ego = torch.load(save_path_ego+f'_best_iter{iteration-1}.pth', weights_only=True)
            policy_br_ego.load_state_dict(checkpoint_ego['model_state_dict'])
            checkpoint_opp = torch.load(save_path_opp+f'_best_iter{iteration-1}.pth', weights_only=True)
            policy_br_opp.load_state_dict(checkpoint_opp['model_state_dict'])

        if whose_turn == 'ego':
            policy_opp_set = self.policy_avg_opp_set
            policy_ego_set = policy_br_ego
        else:
            policy_opp_set = policy_br_opp
            policy_ego_set = self.policy_avg_ego_set
        
        # optimizers for best response policy training
        self.br_optimizer_ego = torch.optim.Adam(policy_br_ego.parameters(), lr=args.lr)
        self.br_optimizer_opp = torch.optim.Adam(policy_br_opp.parameters(), lr=args.lr)

        # evaluate <ego current average policy versus opp current average policy>
        loss_ego, stl_loss, ref_loss, _, stl_roustness, loss_opp = self.rollout(
                                                                args=args,
                                                                policy_ego_set=self.policy_avg_ego_set,
                                                                policy_opp_set=self.policy_avg_opp_set,
                                                                whose_turn=None,
                                                                policy_sample_dist=self.avg_policy_sample_dist,
                                                                )
        stl_roustness_init = stl_roustness.mean()
        print(f"Iteration {iteration}, initial stl robustness {stl_roustness_init}")

        stl_robustness_max, stl_robustness_min = -1e3, 1e3 # initialization for max/min robustness values, for exploitabilty compute
        
        # train current best response
        for epoch in range(args.train_epoch):

            loss_ego, stl_loss, ref_loss, _, stl_roustness, loss_opp = self.rollout(
                                                                            args=args,
                                                                            policy_ego_set=policy_ego_set,
                                                                            policy_opp_set=policy_opp_set,
                                                                            whose_turn=whose_turn,
                                                                            policy_sample_dist=self.avg_policy_sample_dist,
                                                                            )

            loss = loss_ego.mean() if whose_turn == 'ego' else loss_opp.mean()
            loss.backward()
            if epoch % 50 == 0:
                print(f"Iteration {iteration}, agent {whose_turn}, epoch {epoch}, stl loss {stl_loss.mean():.3f}, ref loss {ref_loss.mean():.3f}, stl robustness {stl_roustness.mean():.3f}")
            
            stl_robustness_max = max(stl_roustness.mean(), stl_robustness_max)
            stl_robustness_min = min(stl_roustness.mean(), stl_robustness_min)
                
            if whose_turn == 'ego':
                self.br_optimizer_ego.step()
                self.br_optimizer_ego.zero_grad()
            else:
                self.br_optimizer_opp.step()
                self.br_optimizer_opp.zero_grad()

        # update avg policy set and sample distribution
        if whose_turn == 'opp':
            self.policy_avg_ego_set.append(policy_br_ego)
            self.policy_avg_opp_set.append(policy_br_opp)
            current_time = iteration + 2
            avg_policy_prob = (current_time - 1) / (current_time + 1)
            latest_policy_prob = 2 / (current_time + 1)
            self.avg_policy_sample_dist.append(1 / avg_policy_prob * latest_policy_prob)
            self.avg_policy_sample_dist = [float(i)/sum(self.avg_policy_sample_dist) for i in self.avg_policy_sample_dist]

        # save models
        if whose_turn == 'ego':
            # save best response model at this iteration
            torch.save({
            'model_state_dict': policy_br_ego.state_dict(),
            'optimizer_state_dict': self.br_optimizer_ego.state_dict(),
            }, save_path_ego+f'_best_iter{iteration}.pth')
        
        else:
            # save best response model at this iteration
            torch.save({
            'model_state_dict': policy_br_opp.state_dict(),
            'optimizer_state_dict': self.br_optimizer_opp.state_dict(),
            }, save_path_opp+f'_best_iter{iteration}.pth')

        # compute exploitability
        if whose_turn == 'ego':
            exp_opp = stl_robustness_max - stl_roustness_init
            self.exp_opp = exp_opp
            return None, None
        else:
            exp_ego = stl_roustness_init - stl_robustness_min
            exploitability = exp_ego + self.exp_opp
            return exploitability, self.avg_policy_sample_dist

    # rollout the system using policy_ego_set and policy_opp_set, get robustness, loss, etc
    def rollout(self, args, policy_ego_set, policy_opp_set, whose_turn, policy_sample_dist):
        
        iter_num = len(policy_opp_set) if whose_turn == 'ego' else len(policy_ego_set)
        batch_size = min(2*iter_num + 2, 15) # rollouts we want to sample; more samples lead to more train/eval time

        for batch_index in range(batch_size):
            if whose_turn == 'ego':
                policy_opp = np.random.choice(policy_opp_set, p=policy_sample_dist)
                policy_ego = policy_ego_set
            elif whose_turn == 'opp':
                policy_ego = np.random.choice(policy_ego_set, p=policy_sample_dist)
                policy_opp = policy_opp_set
            else:
                policy_opp = np.random.choice(policy_opp_set, p=policy_sample_dist)
                policy_ego = np.random.choice(policy_ego_set, p=policy_sample_dist)
                
            current_traj_idx, _, _ = args.batch_data_gen_fn(
                                                            args=args,
                                                            init_state_ego=self.ref_X_ego[0:1, :],
                                                            init_state_opp=self.ref_X_opp[0:1, :],
                                                            policy_ego=policy_ego,
                                                            policy_opp=policy_opp,
                                                            )
            if batch_index == 0:
                current_traj = current_traj_idx
            else:
                current_traj = torch.cat((current_traj, current_traj_idx), dim=0)

        # current_traj_ego, control_seq_ego, log_prob_seq_ego = current_traj[..., :state_dim], control_seq[..., :control_dim], log_prob_seq[..., :control_dim]
        # current_traj_opp, control_seq_opp, log_prob_seq_opp = current_traj[..., state_dim:], control_seq[..., control_dim:], log_prob_seq[..., control_dim:]
        current_traj_ego = current_traj[..., :args.state_dim]
        current_traj_opp = current_traj[..., args.state_dim:]

        loss_fn_ego = deterministic_loss_fn_multi_agent_ego
        loss_ego, stl_loss, ref_loss, control_loss, stl_roustness = loss_fn_ego(
            args, self.stl_class, current_traj_ego[..., :args.traj_dim], current_traj_opp[..., :args.traj_dim])
        
        loss_fn_opp = deterministic_loss_fn_multi_agent_opp
        loss_opp, _, _, _, _ = loss_fn_opp(
            args, self.stl_class, current_traj_ego[..., :args.traj_dim], current_traj_opp[..., :args.traj_dim])
        
        return loss_ego, stl_loss, ref_loss, control_loss, stl_roustness, loss_opp