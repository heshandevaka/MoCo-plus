import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class MoCo(AbsWeighting):
    r"""Multiple Objective Gradient Correction (MoCo).
    
    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/pdf?id=dLAYGdKTi2>`

    Args:
        beta_moco (float, default=0.1): Learning rate of tracking sequence.
        gamma_moco (float, default=0.1): Learning rate of lambda sequence.
        rho_moco (float, default=0.1): Regularization parameter of lambda subproblem.
        moco_gn ({'none', 'l2', 'loss', 'loss+'}, default='none'): The type of gradient normalization.

    """
    def __init__(self):
        super(MoCo, self).__init__()

    def init_param(self):
        # initialize Y sequence
        self.y = 0
        # initialize lambda sequence
        self.lambd = 1/self.task_num*torch.ones([self.task_num, ]).to(self.device)

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

    def _next_point(self, cur_val, grad, n):
        proj_grad = grad - ( torch.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])

        skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
        t = torch.ones(1).to(grad.device)
        if (tm1>1e-7).sum() > 0:
            t = torch.min(tm1[tm1>1e-7])
        if (tm2>1e-7).sum() > 0:
            t = torch.min(t, torch.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = self._projection2simplex(next_point)
        return next_point   
    
    def _find_min_norm_element(self, grads):

        def _min_norm_element_from2(v1v1, v1v2, v2v2):
            if v1v2 >= v1v1:
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
            cost = v2v2 + gamma*(v1v2 - v2v2)
            return gamma, cost

        def _min_norm_2d(grad_mat):
            dmin = 1e8
            for i in range(grad_mat.size()[0]):
                for j in range(i+1, grad_mat.size()[0]):
                    c,d = _min_norm_element_from2(grad_mat[i,i], grad_mat[i,j], grad_mat[j,j])
                    if d < dmin:
                        dmin = d
                        sol = [(i,j),c,d]
            return sol

        MAX_ITER = 250
        STOP_CRIT = 1e-5
    
        grad_mat = grads.mm(grads.t())
        init_sol = _min_norm_2d(grad_mat)
        
        n = grads.size()[0]
        sol_vec = torch.zeros(n).to(grads.device)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec
    
        iter_count = 0

        while iter_count < MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = self._next_point(sol_vec, grad_dir, n)

            v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*sol_vec.unsqueeze(0).repeat(n, 1)*grad_mat)
            v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
            v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
    
            nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < STOP_CRIT:
                return sol_vec
            sol_vec = new_sol_vec
    
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
    
    def backward(self, losses, **kwargs):
        # get algo params
        beta_moco = kwargs['beta_moco']
        gamma_moco = kwargs['gamma_moco']
        rho_moco = kwargs['rho_moco']
        moco_gn = kwargs['moco_gn']
        # get gradients from losses
        grads = self._get_grads(losses, mode='backward')

        if self.rep_grad: # assuming False (for now)
            per_grads, grads = grads[0], grads[1]

        # update y sequence
        self.y = self.y - beta_moco * ( self.y - grads )
        
        # get loss values for normalizing
        loss_data = torch.tensor([loss.item() for loss in losses]).to(self.device)

        # normalize y before update lambda (if needed)
        y = self._gradient_normalizers(self.y, loss_data, ntype=moco_gn) # l2, loss, loss+, none

        # print('lambda device:', self.lambd.shape, 'y shape:', self.y.shape) # REMOVE

        # lambda update
        self.lambd = self._projection2simplex( self.lambd - gamma_moco * ( y @ (torch.transpose(y, 0, 1) @ self.lambd ) + rho_moco * self.lambd ) )
        # if exactly solving (not used, at least now): self.lambda = self._find_min_norm_element(y)

        # update x (model) with Y@lambd
        if self.rep_grad: # assuming False (for now)
            self._backward_new_grads(self.lambd, per_grads=per_grads)
        else:
            self._backward_new_grads(self.lambd, grads=y)

        return self.lambd.detach().cpu().numpy()