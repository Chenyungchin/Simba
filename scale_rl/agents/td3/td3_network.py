import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim) # See if this is redundant

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x + residual

class Actor_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_TD3, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        n = 2
        self.blocks = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l2 = nn.Linear(256, action_dim)
        self.layer_norm1 = nn.LayerNorm(action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        for block in self.blocks:
            a = block(a)
        a = F.relu(self.l2(a))
        a = self.layer_norm1(a)
        return self.max_action * torch.tanh(self.l3(a))


class Critic_TD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_TD3, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1