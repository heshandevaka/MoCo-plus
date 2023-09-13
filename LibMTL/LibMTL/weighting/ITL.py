import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class ITL(AbsWeighting):
    r"""Independent Task Learning (ITL).

    Learn one predefined task, independent of other task. Thus, behave as a dedicated single task learner.

    """
    def __init__(self):
        super(ITL, self).__init__()
        
    def backward(self, losses, **kwargs):
        # get task index
        task_idx = kwargs['task_idx']
        # loss considered is the individula task loss
        loss = losses[task_idx]
        loss.backward()
        # creata one hot vector for task
        lambd = np.zeros(self.task_num)
        lambd[task_idx] = 1
        return lambd