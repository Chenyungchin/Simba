import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_rl.agents.running_normal import RunningNorm
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-6

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x + residual

class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_RSNorm=False, use_LayerNorm=False, use_Residual=False):
        super(Actor_SAC, self).__init__()
        self.running_norm = RunningNorm([state_dim])
        self.use_RSNorm = use_RSNorm
        self.use_LayerNorm = use_LayerNorm
        self.use_Residual = use_Residual
        self.hidden_dim = 128

        self.l1 = nn.Linear(state_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Add layer normalization here (for pre norm)
        n = 1
        self.blocks = nn.ModuleList(ResidualBlock(self.hidden_dim) for _ in range(n))
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        # separate mean and log_std
        self.l3 = nn.Linear(self.hidden_dim, action_dim)
        self.l4 = nn.Linear(self.hidden_dim, action_dim)
        self.action_dim = action_dim
        self.max_action = max_action

        print(f"Actor_SAC: use_RSNorm: {self.use_RSNorm}, use_LayerNorm: {self.use_LayerNorm}, use_Residual: {self.use_Residual}")

    def forward(self, state):
        # running norm
        if self.use_RSNorm:
            state = self.running_norm(state)
        # linear layer without relu
        x = self.l1(state)
        # residual
        if self.use_Residual:
            for block in self.blocks:
                x = block(x)
        # layer norm
        if self.use_LayerNorm:
            x = self.layer_norm2(x)
        
        mean = self.l3(x)
        log_std = self.l4(x)
        # mean, log_std = torch.split(x, self.action_dim, dim=-1)
        # log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1) # what simba does for log_std norm
        return mean, log_std

    def sample(self, state):
        if self.use_RSNorm:
            state = self.running_norm(state)

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # debug
        if (torch.isnan(log_prob).any() == True):
            print(f"normal log prob: {normal.log_prob(z)}")
            print(f"log term: {torch.log(1 - action.pow(2) + 1e-6)}")
        return action, log_prob

class Critic_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, use_RSNorm=False, use_LayerNorm=False, use_Residual=False):
        super(Critic_SAC, self).__init__()
        self.state_norm = RunningNorm([state_dim])
        self.action_norm = RunningNorm([action_dim])

        self.use_RSNorm = use_RSNorm
        self.use_LayerNorm = use_LayerNorm
        self.use_Residual = use_Residual

        self.hidden_dim = 512

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_dim)
        n = 2
        self.blocks = nn.ModuleList(ResidualBlock(self.hidden_dim) for _ in range(n))
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_dim)
        self.blocks2 = nn.ModuleList(ResidualBlock(self.hidden_dim) for _ in range(n))
        self.l5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm3 = nn.LayerNorm(self.hidden_dim)
        self.l6 = nn.Linear(self.hidden_dim, 1)

        print(f"Critic_SAC: use_RSNorm: {self.use_RSNorm}, use_LayerNorm: {self.use_LayerNorm}, use_Residual: {self.use_Residual}")


    def forward(self, state, action):
        # running norm
        if self.use_RSNorm:
            state = self.state_norm(state)
            action = self.action_norm(action)
        sa = torch.cat([state, action], 1)
        # linear layer without relu
        q1 = self.l1(sa)
        # residual
        if self.use_Residual:
            for block in self.blocks:
                q1 = block(q1)
        
        # q1 = self.l2(q1)
        # layer norm
        if self.use_LayerNorm:
            q1 = self.layer_norm2(q1)
        q1 = self.l3(q1)
        
        # linear layer without relu
        q2 = self.l4(sa)
        # residual
        if self.use_Residual:
            for block in self.blocks2:
                q2 = block(q2)
        # q2 = self.l5(q2)
        # layer norm
        if self.use_LayerNorm:
            q2 = self.layer_norm3(q2)
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        # running norm
        if self.use_RSNorm:
            state = self.state_norm(state)
            action = self.action_norm(action)
        sa = torch.cat([state, action], 1)
        # linear layer without relu
        q1 = self.l1(sa)
        # residual
        if self.use_Residual:
            for block in self.blocks:
                q1 = block(q1)
        # q1 = F.relu(self.l2(q1))
        # layer norm
        if self.use_LayerNorm:
            q1 = self.layer_norm2(q1)
        q1 = self.l3(q1)
        return q1