import torch
torch.random.manual_seed(0)

# control the velocity in x-y dimensions
def single_integrator(state, control, batch_first=False, state_dim=2, control_dim=2):

    delta_t = 0.1
    A = torch.eye(2)
    B = torch.eye(2) * delta_t
    if not batch_first:
        next_state = A @ state.reshape(state_dim,1) + B @ control[-1,:].reshape(control_dim, 1)
        return next_state.reshape(1, state_dim)
    else:
        batch_size = state.shape[0]
        next_state = A @ state.reshape(batch_size, state_dim, 1) + B @ control.reshape(batch_size, control_dim, 1)
        return next_state.reshape(batch_size, 1, state_dim)

# control the accelaration in x-y dimensions
def double_integrator(state, control, batch_first=False, state_dim=4, control_dim=2):

    delta_t = 0.1
    A = torch.eye(4)
    A[:2, 2:4] = torch.eye(2) * delta_t
    B = torch.cat((torch.zeros(2, 2), delta_t*torch.eye(2)), dim=0)
    if not batch_first:
        next_state = A @ state.reshape(state_dim,1) + B @ control[-1,:].reshape(control_dim, 1)
        return next_state.reshape(1, state_dim)
    else:
        batch_size = state.shape[0]
        next_state = A @ state.reshape(batch_size, state_dim, 1) + B @ control.reshape(batch_size, control_dim, 1)
        return next_state.reshape(batch_size, 1, state_dim)


# kinematic single track racing car model
def kinematic_model(state, control, batch_first=False, state_dim=5, control_dim=2):

    # control: steering velocity + acc

    # discretization time
    delta_t = 0.1

    # vehicle parameters
    lf = 0.3048*3.793293
    lr = 0.3048*4.667707
    lwb = lf + lr

    batch_size = state.shape[0]
    state = state.view(batch_size, state_dim)
    control = control.view(batch_size, control_dim)
    
    # kinematic model
    xdot = state[:,3]*torch.cos(state[:,4]).reshape(batch_size, 1)
    xdot = torch.cat((xdot, (state[:,3]*torch.sin(state[:,4])).reshape(batch_size, 1)), 1)
    xdot = torch.cat((xdot, control[:,0].reshape(batch_size, 1)), 1)
    xdot = torch.cat((xdot, control[:,1].reshape(batch_size, 1)), 1)
    xdot =  torch.cat((xdot, (state[:,3]/lwb*torch.tan(state[:,2])).reshape(batch_size, 1)), 1)
    
    # euler integration
    next_state = state + xdot * delta_t

    return next_state.reshape(batch_size, 1, state_dim)

# control the accelaration in x-y-z dimensions
def double_integrator_3d(state, control, batch_first=False, state_dim=6, control_dim=3):

    delta_t = 0.1
    eye = torch.eye(state_dim//2)
    A = torch.block_diag(eye, eye)
    A[:state_dim//2, state_dim//2:] = delta_t*eye
    B = torch.cat((torch.zeros(state_dim//2, control_dim), delta_t*torch.eye(control_dim)), dim=0)
    if not batch_first:
        next_state = A @ state.reshape(state_dim,1) + B @ control[-1,:].reshape(control_dim, 1)
        return next_state.reshape(1, state_dim)
    else:
        batch_size = state.shape[0]
        next_state = A @ state.reshape(batch_size, state_dim, 1) + B @ control.reshape(batch_size, control_dim, 1)
        return next_state.reshape(batch_size, 1, state_dim)

# control the accelaration in x-y-z dimensions
def quadrotor(state, control, batch_first=False, state_dim=6, control_dim=3):

    A = torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0.2, 0, 0, 1, 0, 0],
                    [0, 0.2, 0, 0, 1, 0],
                    [0, 0, 0.2, 0, 0, 1]
                    ])

    B = torch.tensor([
                    [1.96, 0, 0],
                    [0, -1.96, 0],
                    [0, 0, 0.4],
                    [0.196, 0, 0],
                    [0, -0.196, 0],
                    [0, 0, 0.04]
                    ])

    if not batch_first:
        next_state = A @ state.reshape(state_dim,1) + B @ control[-1,:].reshape(control_dim, 1)
        return next_state.reshape(1, state_dim)
    else:
        batch_size = state.shape[0]
        next_state = A @ state.reshape(batch_size, state_dim, 1) + B @ control.reshape(batch_size, control_dim, 1)
        return next_state.reshape(batch_size, 1, state_dim)


# get a batch of trajectories using policy for single agent
def rollout_batch(init_state, policy, batch_size=128, state_dim=2, control_dim=2,
                  batch_first=True, dynamic=None, stochastic_policy=None, total_time_step=50):

    # append other dimensions (e.g., velocity) by 0 if state_dim > 2
    if init_state.shape[1] < state_dim:
        init_state = torch.cat((init_state, torch.zeros(1, state_dim - init_state.shape[1])), dim=1)
    current_state = init_state.repeat(batch_size, 1, 1) # batch_size * 1 * state_dim
    current_traj = current_state # batch_size * state_dim * 1
    control_seq = torch.zeros((batch_size, total_time_step - 1, control_dim)) # batch_size * (total_time_step - 1) * control_dim 
    log_prob_seq = torch.zeros((batch_size, total_time_step - 1, control_dim)) # same dim, store log_prob of the current action

    for time in range(total_time_step - 1):
        action, log_prob = policy.act(current_traj) # batch_size * action_dim
        current_state = dynamic(current_state, action, batch_first=batch_first)
        current_traj = torch.cat((current_traj, current_state), dim=1)
        control_seq[:,time:time+1,:] = action.reshape(batch_size, 1, control_dim)
        if stochastic_policy:
            log_prob_seq[:,time:time+1,:] = log_prob
    
    if not stochastic_policy:
        log_prob_seq = None
        
    return current_traj, control_seq, log_prob_seq

# get a batch of trajectories using policy for two agents (they run at the same time)
def rollout_batch_two_agents(args, init_state_ego, init_state_opp, policy_ego, policy_opp, opp_noise=None):

    if init_state_ego.shape[1] < args.state_dim:
        init_state_ego = torch.cat((init_state_ego, torch.zeros(1, args.state_dim - init_state_ego.shape[1])), dim=1)
        init_state_opp = torch.cat((init_state_opp, torch.zeros(1, args.state_dim - init_state_opp.shape[1])), dim=1)
    init_state = torch.cat((init_state_ego, init_state_opp), dim=1)
    current_state = init_state.repeat(args.batch_size, 1, 1) # batch_size * 1 * (2*state_dim)
    current_traj = current_state # batch_size * state_dim * 1
    control_seq = torch.zeros((args.batch_size, args.total_time_step - 1, args.control_dim*2)) # batch_size * (total_time_step - 1) * control_dim 
    log_prob_seq = torch.zeros((args.batch_size, args.total_time_step - 1, args.control_dim*2)) # same dim, store log_prob of the current action
    
    for time in range(args.total_time_step - 1):
        action_ego, log_prob_ego = policy_ego.act(current_traj) # batch_size * action_dim
        action_opp, log_prob_opp = policy_opp.act(current_traj) # batch_size * action_dim
        if opp_noise is not None:
            action_opp = action_opp + (torch.rand_like(action_opp) - 0.5) * 0.5
        current_state_ego = args.dynamic_ego(current_state[..., :args.state_dim], action_ego, batch_first=args.batch_first)
        current_state_opp = args.dynamic_opp(current_state[..., args.state_dim:], action_opp, batch_first=args.batch_first)
        current_state = torch.cat((current_state_ego, current_state_opp), dim=2)
        current_traj = torch.cat((current_traj, current_state), dim=1)
        control_seq[:,time:time+1,:args.control_dim] = action_ego.reshape(args.batch_size, 1, args.control_dim)
        control_seq[:,time:time+1,args.control_dim:] = action_opp.reshape(args.batch_size, 1, args.control_dim)
        if log_prob_ego is not None:
            log_prob_seq[:,time:time+1,:args.control_dim] = log_prob_ego
        if log_prob_opp is not None:
            log_prob_seq[:,time:time+1,args.control_dim:] = log_prob_opp
    
    return current_traj, control_seq, log_prob_seq
    
# get a batch of trajectories using policy for two agents with values (they run at the same time)
def rollout_batch_two_agents_with_value(
    args, init_state_ego, init_state_opp, policy_ego, policy_opp
):

    if init_state_ego.shape[1] < args.state_dim:
        init_state_ego = torch.cat(
            (init_state_ego, torch.zeros(1, args.state_dim - init_state_ego.shape[1])),
            dim=1,
        )
        init_state_opp = torch.cat(
            (init_state_opp, torch.zeros(1, args.state_dim - init_state_opp.shape[1])),
            dim=1,
        )
    init_state = torch.cat((init_state_ego, init_state_opp), dim=1)
    current_state = init_state.repeat(
        args.batch_size, 1, 1
    )  # batch_size * 1 * (2*state_dim)
    current_traj = current_state  # batch_size * state_dim * 1
    control_seq = torch.zeros(
        (args.batch_size, args.total_time_step - 1, args.control_dim * 2)
    )  # batch_size * (total_time_step - 1) * control_dim
    log_prob_seq = torch.zeros(
        (args.batch_size, args.total_time_step - 1, args.control_dim * 2)
    )  # same dim, store log_prob of the current action
    value_seq = torch.zeros(args.batch_size, args.total_time_step - 1, 2)

    for time in range(args.total_time_step - 1):
        action_ego, log_prob_ego, entropy_ego, value_ego = (
            policy_ego.get_action_and_value(current_traj)
        )  # batch_size * action_dim
        action_opp, log_prob_opp, entropy_opp, value_opp = (
            policy_opp.get_action_and_value(current_traj)
        )  # batch_size * action_dim

        current_state_ego = args.dynamic_ego(
            current_state[..., : args.state_dim],
            action_ego,
            batch_first=args.batch_first,
        )
        current_state_opp = args.dynamic_opp(
            current_state[..., args.state_dim :],
            action_opp,
            batch_first=args.batch_first,
        )

        current_state = torch.cat((current_state_ego, current_state_opp), dim=2)
        current_traj = torch.cat((current_traj, current_state), dim=1)
        control_seq[:, time : time + 1, : args.control_dim] = action_ego.reshape(
            args.batch_size, 1, args.control_dim
        )
        control_seq[:, time : time + 1, args.control_dim :] = action_opp.reshape(
            args.batch_size, 1, args.control_dim
        )
        if log_prob_ego is not None:
            log_prob_seq[:, time : time + 1, : args.control_dim] = log_prob_ego
        if log_prob_opp is not None:
            log_prob_seq[:, time : time + 1, args.control_dim :] = log_prob_opp

        if value_ego is not None:
            value_seq[:, time : time + 1, 0] = value_ego
        if value_opp is not None:
            value_seq[:, time : time + 1, 1] = value_opp

    return current_traj, control_seq, log_prob_seq, value_seq