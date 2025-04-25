from scale_rl.agents.sac.sac_agent import SAC
from scale_rl.agents.td3.td3_agent import TD3

def create_agent(
        policy_name,
        state_dim,
        action_dim,
        max_action,
        discount, 
        tau,
        # for TD3
        policy_noise,
        noise_clip,
        policy_freq,
        use_RSNorm,
        use_LayerNorm,
        use_Residual,
):  
    # TODO: add other agents
    if policy_name == 'SAC':
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            tau=tau,
            # simba param
            use_RSNorm=use_RSNorm,
            use_LayerNorm=use_LayerNorm,
            use_Residual=use_Residual,
        )
    elif policy_name == 'TD3':
        agent = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            # simba param
            use_RSNorm=use_RSNorm,
            use_LayerNorm=use_LayerNorm,
            use_Residual=use_Residual,
        )

    return agent