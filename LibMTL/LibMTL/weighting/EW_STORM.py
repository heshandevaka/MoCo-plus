import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class EW_STORM(AbsWeighting):
    r"""Equal Weighting + STORM (EW_STORM).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self):
        super(EW_STORM, self).__init__()

    def init_param(self):
        # initialize d sequence
        self.d = 0
        self.t = 1
        self.eta = 0
        self.prev_batch = None
        self.prev_grad = 0
        # initialize lambda sequence
        self.lambd = 1/self.task_num*torch.ones([self.task_num, ]).to(self.device)
        
        
    def backward(self, losses, losses_prev_batch, **kwargs):
        # get algo params
        c = kwargs['c_ew_storm']
        k = kwargs['k_ew_storm']
        w = kwargs['w_ew_storm']
        sigma = kwargs['sigma_ew_storm']

        # calculate a
        a = c * self.eta**2

        # calculate current grads
        grads = self._get_grads(losses, mode='backward')

        if self.rep_grad: # assuming False (for now)
            per_grads, grads = grads[0], grads[1]

        # calculate d
        self.d = grads + (1 - a) * (self.d - self.prev_grad)

        # calculate grads for next d update
        self.prev_grads = self._get_grads(losses_prev_batch, mode='backward')

        # calculate storm stepsize
        # self.eta = k/(w + sigma**2 * self.t) ** (1/3)

        # calculate lambda
        self.lambd = torch.ones_like(losses).to(self.device)

        # loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        # loss.backward()

        # update x (model) with Y@lambd
        if self.rep_grad: # assuming False (for now)
            self._backward_new_grads(self.lambd, per_grads=per_grads)
        else:
            self._backward_new_grads(self.lambd, grads=self.d)

        self.t += 1

        return self.lambd.detach().cpu().numpy()