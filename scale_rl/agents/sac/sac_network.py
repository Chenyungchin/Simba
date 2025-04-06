import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2
epsilon = 1e-6

class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_SAC, self).__init__()
        # [HINT] Construct a neural network as the actor. Return its value using forward You need to write down three linear layers.
        # 1. l1: state_dim → 256
        # 2. l2: 256 → 256
        # 3. l3: 256 → mean and log std of the action
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)
        self.action_dim = action_dim
        ############################
        self.max_action = max_action

    def forward(self, state):
        # [HINT] Use the three linear layers to compute the mean and log std of the action
        # Apply ReLU activation after layer l1 and l2
        ############################
        # YOUR IMPLEMENTATION HERE #
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
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
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        ############################

        # Q2 architecture
        # [HINT] Construct a neural network as the second critic. Return its value using forward. You need to write down three linear layers.
        # 1. l4: state_dim+action_dim → 256
        # 2. l5: 256 → 256
        # 3. l6: 256 → 1
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
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
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        q1 = self.l3(x)

        x = F.relu(self.l4(sa))
        x = F.relu(self.l5(x))
        q2 = self.l6(x)
        ############################
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        # [HINT] only returns q1 for actor update using layers l1, l2, l3
        # 1. Apply ReLU activation after layer l1
        # 2. Apply ReLU activation after layer l2
        # 3. Return output as q1 from layer l3
        ############################
        # YOUR IMPLEMENTATION HERE #
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        q1 = self.l3(x)
        ############################
        return q1