import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from scale_rl.agents.sac.sac_network import (
    Actor_SAC,
    Critic_SAC
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        # simba param
        use_RSNorm=False,
        use_LayerNorm=False,
        use_Residual=False,
        use_MLP_ReLU=False,
        # hyperparam
        lr=3e-4,
        weight_decay=0.0,
    ):
        self.actor = Actor_SAC(
            state_dim, 
            action_dim, 
            max_action,
            use_RSNorm=use_RSNorm,
            use_LayerNorm=use_LayerNorm,
            use_Residual=use_Residual,
            use_MLP_ReLU=use_MLP_ReLU,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
        )

        self.critic = Critic_SAC(
            state_dim, 
            action_dim,
            use_RSNorm=use_RSNorm,
            use_LayerNorm=use_LayerNorm,
            use_Residual=use_Residual,
            use_MLP_ReLU=use_MLP_ReLU,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
        )

        print(f"lr: {lr}")
        print(f"weight_decay: {weight_decay}")

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # if not evaluate:
        #     action, _, _ = self.actor.sample(state)
        # else:
        #     _, _, action = self.actor.sample(state)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # [HINT] compute the target Q value
        # 1. Sample the next action and its log probability from the actor with next_state
        # 2. Compute the next Q values (Q1 and Q2) using the critic_target with next_state and next_action
        # 3. Min over the Q values: target_Q = min(Q1, Q2) - log_prob(a'|s') * alpha
        # 4. Compute the target Q value: target_Q = reward + not_done * discount * target_Q

        ############################
        # YOUR IMPLEMENTATION HERE #
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            # print(f"next_action: {next_action}")
            # print(f"next_log_prob: {next_log_prob}")
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) # TODO
        # print(f"current Q1: {current_Q1}")
        # print(f"current Q2: {current_Q2}")
        # print(f"target Q: {target_Q}")
        # print(f"critic_loss: {critic_loss}")
        ############################

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # [HINT] compute the actor loss
        # 1. Sample the action and its log probability from the actor with state
        # 2. Compute the Q values (Q1 and Q2) using the critic with state and action
        # 3. Min over the Q values: Q = min(Q1, Q2)
        # 4. Compute the actor loss: actor_loss = alpha * log_prob(a|s) - Q

        ############################
        # YOUR IMPLEMENTATION HERE #
        sampled_action, log_prob = self.actor.sample(state)
        Q1, Q2 = self.critic(state, sampled_action)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_prob - Q).mean()
        # print(f"actor loss: {actor_loss}")
        ############################

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)