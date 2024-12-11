import torch
import numpy as np
import random
import copy
from .dynamic_helper import rollout_batch_two_agents, rollout_batch_two_agents_with_value

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
    

class TrainerReccurentPPO:

    def __init__(
        self,
        args,
        stl_class,
        policy_avg_ego=None,
        policy_avg_opp=None,
        p_sample_dist=None,
        policy_br_ego=None,
        policy_br_opp=None,
    ):
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

        self.global_step = 0

    def train(
        self,
        args,
        exp_policy,
        save_path_ego=None,
        save_path_opp=None,
        whose_turn="ego",
        iteration=0,
    ):

        if iteration == 0:
            policy_br_ego = self.policy_br_ego
            policy_br_opp = self.policy_br_opp
        else:
            # load the best response model from last iteration for warmstarting
            policy_br_ego = copy.deepcopy(self.policy_br_ego)
            policy_br_opp = copy.deepcopy(self.policy_br_opp)
            checkpoint_ego = torch.load(
                save_path_ego + f"_best_iter{iteration-1}.pth", weights_only=True
            )
            policy_br_ego.load_state_dict(checkpoint_ego["model_state_dict"])
            checkpoint_opp = torch.load(
                save_path_opp + f"_best_iter{iteration-1}.pth", weights_only=True
            )
            policy_br_opp.load_state_dict(checkpoint_opp["model_state_dict"])

        if whose_turn == "ego":
            policy_opp_set = self.policy_avg_opp_set
            policy_ego_set = policy_br_ego
        else:
            policy_opp_set = policy_br_opp
            policy_ego_set = self.policy_avg_ego_set

        # optimizers for best response policy training
        self.br_optimizer_ego = torch.optim.Adam(policy_br_ego.parameters(), lr=args.lr)
        self.br_optimizer_opp = torch.optim.Adam(policy_br_opp.parameters(), lr=args.lr)

        # evaluate <ego current average policy versus opp current average policy>
        loss_ego, stl_loss, ref_loss, _, stl_robustness, loss_opp, _, _, _, _ = (
            self.rollout(
                args=args,
                policy_ego_set=self.policy_avg_ego_set,
                policy_opp_set=self.policy_avg_opp_set,
                whose_turn=None,
                policy_sample_dist=self.avg_policy_sample_dist,
            )
        )

        stl_robustness_init = stl_robustness.mean()
        print(f"Iteration {iteration}, initial stl robustness {stl_robustness_init}")

        stl_robustness_max, stl_robustness_min = (
            -1e3,
            1e3,
        )  # initialization for max/min robustness values, for exploitabilty compute

        # storage
        num_steps = (args.total_time_step - 1) * args.num_samples
        # all_obs = torch.zeros((num_steps, args.state_dim))
        # all_actions = torch.zeros((num_steps - args.num_samples, args.control_dim))
        # all_logprobs = torch.zeros((num_steps - args.num_samples, args.control_dim))
        # all_rewards = torch.zeros((num_steps - args.num_samples, 1))
        # all_dones = torch.zeros((num_steps - args.num_samples, 1))
        # all_values = torch.zeros((num_steps - args.num_samples, 1))

        # train current best response
        for epoch in range(args.train_epoch):

            if args.anneal_lr:
                frac = 1.0 - (epoch - 1.0) / args.train_epoch
                lrnow = frac * args.lr
                if whose_turn == "ego":
                    self.br_optimizer_ego.param_groups[0]["lr"] = lrnow
                elif whose_turn == "opp":
                    self.br_optimizer_opp.param_groups[0]["lr"] = lrnow

            self.global_step += args.total_time_step

            with torch.no_grad():
                _, _, _, _, stl_robustness, _, traj, u, logprob, values = self.rollout(
                    args,
                    policy_ego_set,
                    policy_opp_set,
                    whose_turn,
                    self.avg_policy_sample_dist,
                )
            bs, step_size, _ = traj.shape

            # current_train_traj = (
            #     traj[..., : args.state_dim]
            #     if whose_turn == "ego"
            #     else traj[..., args.state_dim :]
            # )
            current_train_u = (
                u[..., : args.control_dim]
                if whose_turn == "ego"
                else u[..., args.control_dim :]
            )
            current_train_logprob = (
                logprob[..., : args.control_dim]
                if whose_turn == "ego"
                else logprob[..., args.control_dim :]
            )
            current_train_values = (
                values[..., 0] if whose_turn == "ego" else values[..., 1]
            )

            rewards = torch.zeros(bs, step_size, 1)
            dones = torch.zeros(bs, step_size, 1)

            rewards[:, -1, 0] = (
                stl_robustness if whose_turn == "ego" else -stl_robustness
            )
            dones[:, -1, 0] = 1

            all_obs = traj.reshape((-1, args.state_dim * 2))
            all_rewards = rewards.flatten()
            all_actions = current_train_u.reshape((-1, args.control_dim))
            all_dones = dones.flatten()
            all_logprobs = current_train_logprob.reshape((-1, args.control_dim))
            all_values = current_train_values.flatten()

            with torch.no_grad():
                advantages = torch.zeros_like(all_rewards)
                lastgaelam = 0
                # TODO: check if this is necessary since we know last rollout is done
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - all_dones[t]
                        nextvalues = all_values[t]
                    else:
                        nextnonterminal = 1.0 - all_dones[t]
                        nextvalues = all_values[t + 1]
                    delta = (
                        all_rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - all_values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + all_values

            # flatten batch
            # b_obs = all_obs.view(-1, args.state_dim)
            # b_logprobs = all_logprobs.view(-1, 1)
            # b_actions = all_actions.view(-1, args.control_dim)
            # b_advantages = advantages.view(-1, 1)
            # b_returns = returns.view(-1, 1)
            # b_values = all_values.view(-1, 1)

            b_obs = all_obs
            b_logprobs = all_logprobs
            b_actions = all_actions
            b_advantages = advantages
            b_returns = returns
            b_values = all_values

            # optimize policy and value network
            b_inds = np.arange(num_steps)
            clipfracs = []

            ppo_batch_size = num_steps
            ppo_minibatch_size = ppo_batch_size // args.num_minibatches
            for ppo_epoch in range(args.ppo_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, ppo_batch_size, ppo_minibatch_size):

                    end = start + ppo_minibatch_size
                    mb_inds = b_inds[start:end]

                    if whose_turn == "ego":
                        _, newlogprob, entropy, newvalue = (
                            policy_br_ego.get_action_and_value(
                                b_obs[mb_inds], action=b_actions[mb_inds]
                            )
                        )
                    elif whose_turn == "opp":
                        _, newlogprob, entropy, newvalue = (
                            policy_br_opp.get_action_and_value(
                                b_obs[mb_inds], action=b_actions[mb_inds]
                            )
                        )
                    logratio = newlogprob - b_logprobs[mb_inds].mean(dim=1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    newvalue = newvalue.view(-1, 1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    if whose_turn == "ego":
                        self.br_optimizer_ego.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            policy_br_ego.parameters(), args.max_grad_norm
                        )
                        self.br_optimizer_ego.step()
                    elif whose_turn == "opp":
                        self.br_optimizer_opp.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            policy_br_opp.parameters(), args.max_grad_norm
                        )
                        self.br_optimizer_opp.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.numpy(), b_returns.numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TODO: log here if needed
            wandb.log(
                {
                    f"{iteration}_logs/value_loss": v_loss.item(),
                    f"{iteration}_logs/pg_loss": pg_loss.item(),
                    f"{iteration}_logs/entropy_loss": entropy_loss.item(),
                    f"{iteration}_logs/loss": loss.item(),
                    f"{iteration}_logs/explained_variance": explained_var,
                    f"{iteration}_logs/clipfrac": np.mean(clipfracs),
                    f"{iteration}_logs/stl_robustness": stl_robustness.mean(),
                },
                step=self.global_step,
            )

            if epoch % 50 == 0:
                print(
                    f"Iteration {iteration}, agent {whose_turn}, epoch {epoch}, stl robustness {stl_robustness.mean():.3f}"
                )

            stl_robustness_max = max(stl_robustness.mean(), stl_robustness_max)
            stl_robustness_min = min(stl_robustness.mean(), stl_robustness_min)

            if whose_turn == "ego":
                self.br_optimizer_ego.step()
                self.br_optimizer_ego.zero_grad()
            else:
                self.br_optimizer_opp.step()
                self.br_optimizer_opp.zero_grad()

        # update avg policy set and sample distribution
        if whose_turn == "opp":
            self.policy_avg_ego_set.append(policy_br_ego)
            self.policy_avg_opp_set.append(policy_br_opp)
            current_time = iteration + 2
            avg_policy_prob = (current_time - 1) / (current_time + 1)
            latest_policy_prob = 2 / (current_time + 1)
            self.avg_policy_sample_dist.append(1 / avg_policy_prob * latest_policy_prob)
            self.avg_policy_sample_dist = [
                float(i) / sum(self.avg_policy_sample_dist)
                for i in self.avg_policy_sample_dist
            ]

        # save models
        if whose_turn == "ego":
            # save best response model at this iteration
            torch.save(
                {
                    "model_state_dict": policy_br_ego.state_dict(),
                    "optimizer_state_dict": self.br_optimizer_ego.state_dict(),
                },
                save_path_ego + f"_best_iter{iteration}.pth",
            )

        else:
            # save best response model at this iteration
            torch.save(
                {
                    "model_state_dict": policy_br_opp.state_dict(),
                    "optimizer_state_dict": self.br_optimizer_opp.state_dict(),
                },
                save_path_opp + f"_best_iter{iteration}.pth",
            )

        # compute exploitability

        # TODO: train a fixed stlcg policy to get the exploitability
        exp_optim = torch.optim.Adam(exp_policy.parameters(), lr=args.lr)
        for step in range(args.train_epoch):
            loss_ego, stl_loss, ref_loss, _, stl_robustness, loss_opp, _, _, _, _ = (
                self.rollout_against_fixed(args, policy_ego_set, policy_opp_set, exp_policy, whose_turn)
            )
            loss = loss_ego.mean() if whose_turn == "ego" else loss_opp.mean()
            exp_optim.zero_grad()
            loss.backward()
            exp_optim.step()

            if step % 50 == 0:
                print(
                    f"Training EXP policy, iteration {iteration}, agent {whose_turn}, epoch {step}, stl loss {stl_loss.mean():.3f}, ref loss {ref_loss.mean():.3f}, stl robustness {stl_robustness.mean():.3f}"
                )

            stl_robustness_max = max(stl_robustness.mean(), stl_robustness_max)
            stl_robustness_min = min(stl_robustness.mean(), stl_robustness_min)

        # save exp model here
        torch.save(
            {"model_state_dict": exp_policy.state_dict()},
            save_path_ego + f"_exp_iter{iteration}.pth",
        )

        if whose_turn == "ego":
            exp_opp = stl_robustness_max - stl_robustness_init
            self.exp_opp = exp_opp
            return None, None
        else:
            exp_ego = stl_robustness_init - stl_robustness_min
            exploitability = exp_ego + self.exp_opp
            return exploitability, self.avg_policy_sample_dist

    # rollout the system using policy_ego_set and policy_opp_set, get robustness, loss, etc
    def rollout(
        self, args, policy_ego_set, policy_opp_set, whose_turn, policy_sample_dist
    ):

        # iter_num = len(policy_opp_set) if whose_turn == "ego" else len(policy_ego_set)
        # batch_size = min(
        #     2 * iter_num + 2, 15
        # )  # rollouts we want to sample; more samples lead to more train/eval time
        batch_size = args.num_samples

        for batch_index in range(batch_size):
            if whose_turn == "ego":
                policy_opp = np.random.choice(policy_opp_set, p=policy_sample_dist)
                policy_ego = policy_ego_set
            elif whose_turn == "opp":
                policy_ego = np.random.choice(policy_ego_set, p=policy_sample_dist)
                policy_opp = policy_opp_set
            else:
                policy_opp = np.random.choice(policy_opp_set, p=policy_sample_dist)
                policy_ego = np.random.choice(policy_ego_set, p=policy_sample_dist)

            current_traj_idx, us, logprobs, values = (
                rollout_batch_two_agents_with_value(
                    args=args,
                    init_state_ego=self.ref_X_ego[0:1, :],
                    init_state_opp=self.ref_X_opp[0:1, :],
                    policy_ego=policy_ego,
                    policy_opp=policy_opp,
                )
            )
            if batch_index == 0:
                current_traj = current_traj_idx[:, :-1, :]
                current_val = values
                current_logprob = logprobs
                current_action = us
            else:
                current_traj = torch.cat(
                    (current_traj, current_traj_idx[:, :-1, :]), dim=0
                )
                current_val = torch.cat((current_val, values), dim=0)
                current_logprob = torch.cat((current_logprob, logprobs), dim=0)
                current_action = torch.cat((current_action, us), dim=0)

        # current_traj_ego, control_seq_ego, log_prob_seq_ego = current_traj[..., :state_dim], control_seq[..., :control_dim], log_prob_seq[..., :control_dim]
        # current_traj_opp, control_seq_opp, log_prob_seq_opp = current_traj[..., state_dim:], control_seq[..., control_dim:], log_prob_seq[..., control_dim:]
        current_traj_ego = current_traj[..., : args.state_dim]
        current_traj_opp = current_traj[..., args.state_dim :]

        loss_fn_ego = deterministic_loss_fn_multi_agent_ego
        loss_ego, stl_loss, ref_loss, control_loss, stl_roustness = loss_fn_ego(
            args,
            self.stl_class,
            current_traj_ego[..., : args.traj_dim],
            current_traj_opp[..., : args.traj_dim],
        )

        loss_fn_opp = deterministic_loss_fn_multi_agent_opp
        loss_opp, _, _, _, _ = loss_fn_opp(
            args,
            self.stl_class,
            current_traj_ego[..., : args.traj_dim],
            current_traj_opp[..., : args.traj_dim],
        )

        return (
            loss_ego,
            stl_loss,
            ref_loss,
            control_loss,
            stl_roustness,
            loss_opp,
            current_traj,
            current_action,
            current_logprob,
            current_val,
        )

    def rollout_against_fixed(self, args, policy_ego_set, policy_opp_set, stlcg_policy, whose_turn):
        batch_size = args.num_samples
        for batch_index in range(batch_size):
            if whose_turn == "ego":
                policy_opp = np.random.choice(policy_opp_set, p=self.avg_policy_sample_dist)
                policy_ego = stlcg_policy
            elif whose_turn == "opp":
                policy_ego = np.random.choice(policy_ego_set, p=self.avg_policy_sample_dist)
                policy_opp = stlcg_policy

            current_traj_idx, us, logprobs = rollout_batch_two_agents(
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

            current_traj_ego = current_traj_idx[..., : args.state_dim]
            current_traj_opp = current_traj_idx[..., args.state_dim :]

            loss_fn_ego = deterministic_loss_fn_multi_agent_ego
            loss_ego, stl_loss, ref_loss, control_loss, stl_roustness = loss_fn_ego(
                args,
                self.stl_class,
                current_traj_ego[:, :-1, : args.traj_dim],
                current_traj_opp[:, :-1, : args.traj_dim],
            )

            loss_fn_opp = deterministic_loss_fn_multi_agent_opp
            loss_opp, _, _, _, _ = loss_fn_opp(
                args,
                self.stl_class,
                current_traj_ego[:, :-1, : args.traj_dim],
                current_traj_opp[:, :-1, : args.traj_dim],
            )

        return (
            loss_ego,
            stl_loss,
            ref_loss,
            control_loss,
            stl_roustness,
            loss_opp,
            current_traj_idx,
            us,
            logprobs,
            None,
        )