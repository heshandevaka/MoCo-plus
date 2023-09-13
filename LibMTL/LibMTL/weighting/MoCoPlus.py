import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class MoCoPlus(AbsWeighting):
    r"""MoCo+.

    A faster variance reduction extention to MoCo.

    Args:
        c_mocoplus (float, default=0.1):c parameter in MoCo+.
        k_mocoplus (float, default=0.1): k parameter in MoCo+.
        w_mocoplus (float, default=0.1): w parameter in MoCo+.
        sigma_mocoplus (float, default=0.1): variance parameter in MoCo+.
        gamma_mocoplus (float, default=0.1): lambda stepsize for MoCo+.
        gn_mocoplus ({'none', 'l2', 'loss', 'loss+'}, default='none'): y sequence normalization for MoCo+.

    """
    def __init__(self):
        super(MoCoPlus, self).__init__()

    def init_param(self):
        # initialize d sequence
        self.y = 0
        self.t = 1
        self.alpha = 0
        self.prev_batch = None
        self.prev_grad = 0
        # initialize lambda sequence
        self.lambd = 1/self.task_num*torch.ones([self.task_num, ]).to(self.device)
        # for y sequence projection
        self.L = 0 

    def _projection2simplex(self, y):
        m = len(y)
        sorted_y = torch.sort(y, descending=True)[0]
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(m).to(y.device))
    

    def _gradient_normalizers(self, grads, loss_data, ntype):
        if ntype == 'l2':
            gn = grads.pow(2).sum(-1).sqrt()
        elif ntype == 'loss':
            gn = loss_data
        elif ntype == 'loss+':
            gn = loss_data * grads.pow(2).sum(-1).sqrt()
        elif ntype == 'none':
            gn = torch.ones_like(loss_data).to(self.device)
        else:
            raise ValueError('No support normalization type {} for MoCo'.format(ntype))
        grads = grads / gn.unsqueeze(1).repeat(1, grads.size()[1])
        return grads
        
        
    def backward(self, losses, losses_prev_batch, input_grads=False, **kwargs):
        # get algo params
        c = kwargs['c_mocoplus']
        k = kwargs['k_mocoplus']
        w = kwargs['w_mocoplus']
        sigma = kwargs['sigma_mocoplus']
        gamma = kwargs['gamma_mocoplus']
        ntype = kwargs['gn_mocoplus']

        # calculate beta
        beta = c * self.alpha**2

        # calculate current grads
        if input_grads: # inputs are actually pre calculated grads
            grads = losses
        # if isinstance(losses, list):
        #     grads = 0
        #     for i, losses_ in enumerate(losses):
        #         if i+1 != len(losses):
        #             print(f"calculating grad {i} ...")
        #             grads += self._get_grads(losses_, mode='backward', retain_graph=True)
        #             print("done")
        #         else:
        #             grads += self._get_grads(losses_, mode='backward')
        #     grads = grads/len(losses)
        else:
            grads = self._get_grads(losses, mode='backward')

        if self.rep_grad: # assuming False (for now)
            per_grads, grads = grads[0], grads[1]

        # calculate d
        self.L = max(self.L, torch.norm(grads, dim=0).max().item())
        self.y = grads + (1 - beta) * (self.y - self.prev_grad)
        self.y =  self.L * self._gradient_normalizers(self.y, losses, ntype)
        # self.d = self._gradient_normalizers(self.d, losses, ntype)

        # calculate grads for next d update
        if input_grads: # inputs are actually pre calculated grads
            self.prev_grads = losses_prev_batch
        # if isinstance(losses, list):
        #     self.prev_grads = 0
        #     for losses_ in losses_prev_batch:
        #         self.prev_grads += self._get_grads(losses_, mode='backward')
        #     self.prev_grads = self.prev_grads/len(losses)
        else:
            self.prev_grads = self._get_grads(losses_prev_batch, mode='backward')

        # calculate storm stepsize
        self.alpha = k/(w + sigma**2 * self.t) ** (1/3)

        # calculate lambda step size
        # gamma = self.eta/k

        # calculate lambda
        self.lambd = self._projection2simplex( self.lambd - gamma * ( self.y @ (torch.transpose(self.y, 0, 1) @ self.lambd )) )

        # loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        # loss.backward()

        # update x (model) with Y@lambd
        if self.rep_grad: # assuming False (for now)
            self._backward_new_grads(self.lambd, per_grads=per_grads)
        else:
            self._backward_new_grads(self.lambd, grads=self.y)

        self.t += 1

        return self.lambd.detach().cpu().numpy()