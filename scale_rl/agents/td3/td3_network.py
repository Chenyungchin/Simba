import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_rl.agents.running_normal import RunningNorm
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        return x + residual

class Actor_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_SIMBA=True):
        super(Actor_TD3, self).__init__()
        self.running_norm = RunningNorm([state_dim])

        self.use_SIMBA= use_SIMBA

        self.l1 = nn.Linear(state_dim, 256)
        n = 3
        self.blocks = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.layer_norm1 = nn.LayerNorm(256)

        self.max_action = max_action


    def forward(self, state):
        state = self.running_norm(state)
        a = F.relu(self.l1(state))
        if self.use_SIMBA:
            for block in self.blocks:
                a = block(a)
        a = F.relu(self.l2(a))
        if self.use_SIMBA:
            a = self.layer_norm1(a)
        a = self.l3(a)
        return self.max_action * torch.tanh(a)


class Critic_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, use_SIMBA=True):
        super(Critic_TD3, self).__init__()
        self.state_norm = RunningNorm([state_dim])
        self.action_norm = RunningNorm([action_dim])

        self.use_SIMBA = use_SIMBA

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        n = 3
        self.blocks = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l2 = nn.Linear(256, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)


        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.blocks2 = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l5 = nn.Linear(256, 256)
        self.layer_norm3 = nn.LayerNorm(256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        state = self.state_norm(state)
        action = self.action_norm(action)
        sa = torch.cat([state, action], 1)
        
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        if self.use_SIMBA:
            for block in self.blocks:
                q1 = block(q1)
        q1 = F.relu(self.l2(q1))
        if self.use_SIMBA:
            q1 = self.layer_norm2(q1)
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        if self.use_SIMBA:
            for block in self.blocks2:
                q2 = block(q2)
        q2 = F.relu(self.l5(q2))
        if self.use_SIMBA:
            q2 = self.layer_norm3(q2)
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        state = self.state_norm(state)
        action = self.action_norm(action)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        if self.use_SIMBA:
            for block in self.blocks:
                q1 = block(q1)
        q1 = F.relu(self.l2(q1))
        if self.use_SIMBA:
            q1 = self.layer_norm2(q1)
        q1 = self.l3(q1)
        return q1