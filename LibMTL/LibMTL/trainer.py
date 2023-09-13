import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). 
                            The sub-dictionary for each task has four entries whose keywords are named 
                            **metrics**, **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics
                            for this task. # ? m=1 most cases? (edit: m>1 for nyu-v2)
                            The list of **metrics_fn** has two elements, i.e., the updating and score 
                            functions, meaning how to update thoes objectives in the training process and obtain the final 
                            scores, respectively. 
                            The list of **loss_fn** has ``m`` loss functions corresponding to each
                            metric. 
                            The list of **weight** has ``m`` binary integers corresponding to each
                            metric, where ``1`` means the higher the score is, the better the performance,
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. 
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``,
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu' # FIXED hardcoded use of gpu
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)

        # set base results to compute average performance gap
        if list(self.task_dict.keys()) == ['amazon', 'dslr', 'webcam']: # office-31
            self.base_result = {'amazon':0.875, 'dslr':0.9888, 'webcam':0.9732} 
        elif list(self.task_dict.keys()) == ['Art', 'Clipart', 'Product', 'Real_World']: # office-home
            self.base_result = {'Art':0.6698, 'Clipart':0.8202, 'Product':0.9153, 'Real_World':0.8097} 
        elif list(self.task_dict.keys()) == ['segmentation', 'depth', 'normal']: # nyu-v2 
            self.base_result = {'segmentation':np.array([0.5394, 0.7567]), 'depth':np.array([0.3949, 0.1634]), 'normal':np.array([22.1193, 15.4887,  0.3835,  0.643,   0.747])} 
        else:
            self.base_result = None
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input, self.base_result)
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            # batch a sample from data_loader (default 64)
            data, label = next(loader[1]) # CHANGED
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1]) # CHANGED
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):            
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    # double sample if MoDo
                    if self.kwargs['weight_args']['weighting'] == 'MoDo':
                        # init 2 sample collector (different from train_losses for other methods)
                        train_losses = []
                        # collect two independant samples
                        for i in range(2):
                            train_inputs, train_gts = self._process_data(train_loader)
                            train_preds = self.model(train_inputs)
                            train_preds = self.process_preds(train_preds)
                            train_losses_ = self._compute_loss(train_preds, train_gts)
                            train_losses.append(train_losses_.clone())
                            # if i==0:
                            #     train_preds = train_preds_
                            #     train_gts = train_gts_
                            # if i==1:
                            #     for key in list(train_preds.keys()):
                            #         train_preds[key] = torch.cat((train_preds[key], train_preds_[key]), dim=0)
                            #         train_gts[key] = torch.cat((train_gts[key], train_gts_[key]), dim=0)
                        self.meter.update(train_preds, train_gts)
                    
                    elif self.kwargs['weight_args']['weighting'] == 'EW_STORM':
                        if epoch == 0:
                            train_inputs, train_gts = self._process_data(train_loader)
                            train_inputs_, train_gts_ = self._process_data(train_loader)
                            train_preds_ = self.model(train_inputs_)
                            train_preds_ = self.process_preds(train_preds_)
                            train_losses_ = self._compute_loss(train_preds_, train_gts_)
                            train_preds = self.model(train_inputs)
                            train_preds = self.process_preds(train_preds)
                            train_losses = self._compute_loss(train_preds, train_gts)
                            self.model.prev_batch = (train_inputs_, train_gts_ )
                            self.meter.update(train_preds, train_gts)
                        else:
                            train_inputs, train_gts = self.model.prev_batch
                            train_inputs_, train_gts_ = self._process_data(train_loader)
                            train_preds_ = self.model(train_inputs_)
                            train_preds_ = self.process_preds(train_preds_)
                            train_losses_ = self._compute_loss(train_preds_, train_gts_)
                            train_preds = self.model(train_inputs)
                            train_preds = self.process_preds(train_preds)
                            train_losses = self._compute_loss(train_preds, train_gts)
                            self.model.prev_batch = (train_inputs_, train_gts_ )
                            self.meter.update(train_preds, train_gts)

                    else:
                        train_inputs, train_gts = self._process_data(train_loader)
                        train_preds = self.model(train_inputs)
                        train_preds = self.process_preds(train_preds)
                        train_losses = self._compute_loss(train_preds, train_gts)
                        self.meter.update(train_preds, train_gts)
                else:
                    # double sample if MoDo
                    if self.kwargs['weight_args']['weighting'] == 'MoDo':
                        # init 2 sample collector (different from train_losses for other methods)
                        train_losses = []
                        # collect two independant samples
                        for i in range(2):
                            # dummy train_losses_ to be collected in train_losses
                            train_losses_ = torch.zeros(self.task_num).to(self.device)
                            for tn, task in enumerate(self.task_name):
                                train_input, train_gt = self._process_data(train_loader[task])
                                train_pred = self.model(train_input, task)
                                train_pred = train_pred[task]
                                train_pred = self.process_preds(train_pred, task)
                                train_losses_[tn] = self._compute_loss(train_pred, train_gt, task)
                                self.meter.update(train_pred, train_gt, task)
                            # collect the loss sample (clone to be safe)
                            train_losses.append(train_losses_.clone())
                    
                    elif self.kwargs['weight_args']['weighting'] == 'EW_STORM':
                        # init prev batch collector to calcualte gradient with next iterate
                        if epoch == 0:
                            self.model.prev_batch = {}
                        # calculate loss using first batch
                        train_losses = torch.zeros(self.task_num).to(self.device)
                        for tn, task in enumerate(self.task_name):
                            # sampling twice needed in first epoch, else use stored sample
                            if epoch == 0:
                                train_input, train_gt = self._process_data(train_loader[task])
                            else:
                                train_input, train_gt = self.model.prev_batch[task]
                            train_pred = self.model(train_input, task)
                            train_pred = train_pred[task]
                            train_pred = self.process_preds(train_pred, task)
                            train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                            self.meter.update(train_pred, train_gt, task)
                        # calculate loss using second batch
                        train_losses_ = torch.zeros(self.task_num).to(self.device)
                        for tn, task in enumerate(self.task_name):
                            train_input_, train_gt_ = self._process_data(train_loader[task])
                            self.model.prev_batch[task] = (train_input_, train_gt_)
                            train_pred_ = self.model(train_input_, task)
                            train_pred_ = train_pred_[task]
                            train_pred_ = self.process_preds(train_pred_, task)
                            train_losses_[tn] = self._compute_loss(train_pred_, train_gt_, task)

                    else:
                        train_losses = torch.zeros(self.task_num).to(self.device)
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_pred = self.model(train_input, task)
                            train_pred = train_pred[task]
                            train_pred = self.process_preds(train_pred, task)
                            train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                            self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad()
                if self.kwargs['weight_args']['weighting'] == 'EW_STORM':
                    w = self.model.backward(train_losses, train_losses_, **self.kwargs['weight_args'])
                    # for g in self.optim.param_groups:
                    #     g['lr'] = self.model.eta
                    self.model.eta = self.optimizer.param_groups[0]['lr'] 
                else:
                    w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
        # ADDED to manage memory consumption (not needed)
        # del test_loss
        # del test_pred
        # del test_input
        # del test_gt
