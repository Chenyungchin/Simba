import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RunningNorm(nn.Module):
    def __init__(self, shape, epsilon=1e-5, momentum=0.01):
        super(RunningNorm, self).__init__()
        self.register_buffer("running_mean", torch.zeros(*shape))
        self.register_buffer("running_var", torch.ones(*shape))
        # self.register_buffer("count", torch.ones(1))
        self.momentum = momentum
        self.epsilon = epsilon
        self.shape = shape
        
    def forward(self, x, update=True):
        # Use running stats for normalization
        # if x.shape[-1] != self.running_mean.shape[0]:
        #     return x
            
        out = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        
        # Update running stats during training
        if update and self.training:
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)

                # Update running mean and variance
                delta = batch_mean - self.running_mean
                self.running_mean += self.momentum * delta
                self.running_var = (1 - self.momentum) * self.running_var + \
                                self.momentum * batch_var
                
        return out