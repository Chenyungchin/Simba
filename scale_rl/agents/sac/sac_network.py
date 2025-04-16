import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_rl.agents.running_normal import RunningNorm
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-6

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

class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_SAC, self).__init__()
        self.running_norm = RunningNorm([state_dim])
        # [HINT] Construct a neural network as the actor. Return its value using forward You need to write down three linear layers.
        # 1. l1: state_dim → 256
        # 2. l2: 256 → 256
        # 3. l3: 256 → mean and log std of the action
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.l1 = nn.Linear(state_dim, 256)
        # Adding layer normalization here
        
        # Block to be replicated starts here
        n = 2
        self.blocks = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        # Block to be replicated ends here
        self.l2 = nn.Linear(256, 2 * action_dim)
        self.layer_norm2 = nn.LayerNorm(2 * action_dim)
        self.action_dim = action_dim
        ############################
        self.max_action = max_action

    def forward(self, state):
        # [HINT] Use the three linear layers to compute the mean and log std of the action
        # Apply ReLU activation after layer l1 and l2
        ############################
        # YOUR IMPLEMENTATION HERE #
        state = self.running_norm(state)
        
        x = F.relu(self.l1(state))
        # Adding residual connections here
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.l2(x))
        x = self.layer_norm2(x)
        mean, log_std = torch.split(x, self.action_dim, dim=-1)
        ############################
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        # [HINT] Use the forward method to compute the action, its log probability
        # 1. Compute the mean and log std of the action
        # 2. Compute the standard deviation of the action
        # 3. Get the normal distribution of the action
        # 4. Sample the action from the normal distribution
        # 5. Apply tanh to the action and multiply by max_action to ensure the action is in the range of the action space
        # 6. Compute the log probability of the action

        ############################
        # YOUR IMPLEMENTATION HERE #
        state = self.running_norm(state,update=self.training)

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
        ############################
        return action, log_prob

class Critic_SAC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_SAC, self).__init__()
        # Q1 architecture
        # [HINT] Construct a neural network as the first critic. Return its value using forward You need to write down three linear layers.
        # 1. l1: state_dim+action_dim → 256
        # 2. l2: 256 → 256
        # 3. l3: 256 → 1
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.state_norm = RunningNorm([state_dim])
        self.action_norm = RunningNorm([action_dim])

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        n = 2
        self.blocks = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l2 = nn.Linear(256, 1)
        ############################

        # Q2 architecture
        # [HINT] Construct a neural network as the second critic. Return its value using forward. You need to write down three linear layers.
        # 1. l4: state_dim+action_dim → 256
        # 2. l5: 256 → 256
        # 3. l6: 256 → 1
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.l3 = nn.Linear(state_dim + action_dim, 256)
        self.blocks2 = nn.ModuleList(ResidualBlock(256) for _ in range(n))
        self.l4 = nn.Linear(256, 1)
        ############################


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # [HINT] We use layers l1, l2, l3 to obtain q1
        # 1. Apply ReLU activation after layer l1
        # 2. Apply ReLU activation after layer l2
        # 3. Return output as q1 from layer l3

        # [HINT] We use layers l4, l5, l6 to obtain q2
        # 1. Apply ReLU activation after layer l4
        # 2. Apply ReLU activation after layer l5
        # 3. Return output as q2 from layer l6

        ############################
        # YOUR IMPLEMENTATION HERE #
        state = self.state_norm(state)
        action = self.action_norm(action)
        sa = torch.cat([state, action], 1)

        x = F.relu(self.l1(sa))
        for block in self.blocks:
            x = block(x)
        q1 = self.l2(x)

        x = F.relu(self.l3(sa))
        for block in self.blocks2:
            x = block(x)
        q2 = self.l4(x)
        ############################
        return q1, q2


    def Q1(self, state, action):
        # [HINT] only returns q1 for actor update using layers l1, l2, l3
        # 1. Apply ReLU activation after layer l1
        # 2. Apply ReLU activation after layer l2
        # 3. Return output as q1 from layer l3
        ############################
        # YOUR IMPLEMENTATION HERE #
        state = self.state_norm(state, update=self.training)
        action = self.action_norm(action, update=self.training)
        sa = torch.cat([state, action], 1)

        x = F.relu(self.l1(sa))
        for block in self.blocks:
            x = block(x)
        q1 = self.l2(x)
        ############################
        return q1