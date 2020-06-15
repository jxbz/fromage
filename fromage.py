"""
Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import torch
import math
from torch.optim.optimizer import Optimizer


class Fromage(Optimizer):

    def __init__(self, params, lr=0.01, p_bound=None):
        """The Fromage optimiser.
        Arguments:
            lr (float): The learning rate. 0.01 is a good initial value to try.
            p_bound (float): Restricts the optimisation to a bounded set. A
                value of 2.0 restricts parameter norms to lie within 2x their
                initial norms. This regularises the model class.
        """
        self.p_bound = p_bound
        defaults = dict(lr=lr)
        super(Fromage, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 and self.p_bound is not None:
                    state['max'] = self.p_bound*p.norm().item()
                
                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()

                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(-group['lr'], d_p * (p_norm / d_p_norm))
                else:
                    p.data.add_(-group['lr'], d_p)
                p.data /= math.sqrt(1+group['lr']**2)

                if self.p_bound is not None:
                    p_norm = p.norm().item()
                    if p_norm > state['max']:
                        p.data *= state['max']/p_norm

        return loss
