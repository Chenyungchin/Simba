from scale_rl.buffers.replay_buffer import ReplayBuffer

def create_buffer(
        state_dim,
        action_dim,
        max_size=(int(1e6))
):
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=max_size
    )
    return buffer