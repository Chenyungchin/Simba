import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RunningNorm(nn.Module):
    def __init__(self, shape, epsilon=1e-5, init_eps=1e-4):
        """
        Args:
            shape: The shape of the normalized features.
            epsilon: Small constant for numerical stability during normalization.
            init_eps: Initial count (epsilon) for the running statistics.
        """
        super(RunningNorm, self).__init__()
        self.register_buffer("running_mean", torch.zeros(*shape))
        self.register_buffer("running_var", torch.ones(*shape))
        self.register_buffer("count", torch.tensor(init_eps, dtype=torch.float32))
        self.epsilon = epsilon
        self.shape = shape
        
    def forward(self, x, update=True):
        # Normalize using the running statistics.
        out = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        
        # Update running stats during training.
        if update and self.training:
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)
                batch_count = x.shape[0]
                
                # Parallel update of mean, var, and count.
                delta = batch_mean - self.running_mean
                tot_count = self.count + batch_count
                
                # New mean.
                new_mean = self.running_mean + delta * batch_count / tot_count
                
                # Combine the sum of squared differences.
                m_a = self.running_var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
                new_var = M2 / tot_count
                
                # Update buffers.
                self.running_mean.copy_(new_mean)
                self.running_var.copy_(new_var)
                self.count.copy_(tot_count)
                
        return out