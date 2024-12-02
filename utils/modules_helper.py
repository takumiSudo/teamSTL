import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.distributions as distributions
import torch.nn.functional as F

torch.manual_seed(1)

class LSTM(nn.Module):

    def __init__(self, hidden_dim, pred_length, state_dim, use_softmax=False, stochastic_policy = False,
                 num_layers=1, action_dim=2, action_max=1.0, batch_first = False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_length = pred_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_softmax = use_softmax
        self.action_min, self.action_max = -action_max, action_max
        self.stochastic_policy = stochastic_policy
        self.batch_first = batch_first
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = state_dim, hidden_size = hidden_dim, num_layers=num_layers, batch_first=batch_first)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, self.pred_length * action_dim)
        self.std_layer = nn.Linear(hidden_dim, self.pred_length * action_dim)
        self.sm = nn.Softmax(dim=1)

    def forward(self, trajectory):
        if self.stochastic_policy:
            action,log_prob = self.stochastic_forward(trajectory)
            action = torch.clamp(action, min=self.action_min, max=self.action_max)
            return action, log_prob
        else:
            action = self.deterministic_forward(trajectory)
            action = torch.clamp(action, min=self.action_min, max=self.action_max)
            return action, None
        

    def deterministic_forward(self, trajectory):
        if self.batch_first:
            self.batch_size = trajectory.shape[0]
        output, (h_out, _) = self.lstm(trajectory)
        out = self.hidden2out(h_out)
        if self.batch_first:
            out = out.reshape(self.batch_size, self.pred_length, self.action_dim)
        else:
            out = out.reshape(self.pred_length, self.action_dim)
        if self.use_softmax:
            out = self.sm(out)
        return out
    
    def stochastic_forward(self, trajectory, action=None):
        if self.batch_first:
            self.batch_size = trajectory.shape[0]
        output, (h_out, _) = self.lstm(trajectory)
        action_mean = self.hidden2out(h_out)
        std = F.softplus(self.std_layer(h_out))
        if self.batch_first:
            action_mean = action_mean.reshape(self.batch_size, self.pred_length, self.action_dim)
            std = std.reshape(self.batch_size, self.pred_length, self.action_dim)
        else:
            action_mean = action_mean.reshape(self.pred_length, self.action_dim)
            std = std.reshape(self.pred_length, self.action_dim)
        distribution = distributions.Normal(action_mean, std)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action, log_prob

    def act(self, trajectory):
        return self.forward(trajectory)
