import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from create_dataset import office_dataloader

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

import os
import numpy as np

def parse_args(parser):
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--light_model', action='store_true', help='whether to use a resnet encoder or a light linear encoder')
    return parser.parse_args()

def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    if params.dataset == 'office-31':
        task_name = ['amazon', 'dslr', 'webcam']
        class_num = 31
    elif params.dataset == 'office-home':
        task_name = ['Art', 'Clipart', 'Product', 'Real_World']
        class_num = 65
        if params.weighting in ['EW_STORM', 'MoCoPlus']:
            params.bs = params.bs//2
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
    
    # define tasks
    task_dict = {task: {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]} for task in task_name}
    
    # prepare dataloaders
    data_loader, _ = office_dataloader(dataset=params.dataset, batchsize=params.bs, root_path=params.dataset_path, balanced=params.balanced)
    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}
    
    # define encoder and decoders
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            hidden_dim = 512
            self.resnet_network = resnet18(pretrained=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = self.resnet_network(inputs)
            out = torch.flatten(self.avgpool(out), 1)
            out = self.hidden_layer(out)
            return out

    decoders = nn.ModuleDict({task: nn.Linear(512, class_num) for task in list(task_dict.keys())})

    # light weight encoders and decoders to experiment in under-param regime
    class Encoder_Light(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            # size of flattened image for office-31 data
            image_size = 3*224*224
            # hidden dim size (small compared to data size/number of tasks)
            hidden_dim = 2
            # number of classes for office-31
            class_num = 31
            # flatten layer to flatten the input
            self.flatten_layer = nn.Flatten()
            self.hidden_layer = nn.Linear(image_size, hidden_dim)

            # # initialization
            # self.hidden_layer[0].weight.data.normal_(0, 0.005)
            # self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = self.flatten_layer(inputs)
            out = self.hidden_layer(out)
            return out

    decoders_light = nn.ModuleDict({task: nn.Linear(2, class_num) for task in list(task_dict.keys())})

    # handling whether or not to use light model
    model_type='encoder=resnet18_decoders=linear' # use for naming saved models
    if params.light_model:
        # abuse of variable names to make code less bulky
        Encoder = Encoder_Light
        decoders = decoders_light
        model_type = 'light_model_encoder=linear_decoders=linear'

    class Officetrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(Officetrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
            # needed for model save
            self.optim_param = optim_param

        # defined here to modify train method to save models
        def train(self, train_dataloaders, test_dataloaders, epochs, 
                val_dataloaders=None, return_weight=False, save_model=False, save_model_int=2):
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
                save_model (bool): if ``True``, model parameters will be saved every save_model_int epochs
                save_model_int (int): If save_model is ``True``, the inerval of epochs to save the model parameters
            '''
            train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
            train_batch = max(train_batch) if self.multi_input else train_batch
            
            self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
            self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
            self.model.epochs = epochs
            for epoch in range(epochs):

                # save model
                if save_model:
                    if epoch%save_model_int==0:
                        if self.kwargs["weight_args"]["weighting"]=="ITL":
                            model_save_dir = f'./trained_models_{params.dataset}/{self.kwargs["weight_args"]["weighting"]}_task-{self.kwargs["weight_args"]["task_idx"]}_{model_type}_{self.optim_param["optim"]}_lr={self.optim_param["lr"]}_bs={params.bs}_wd={self.optim_param["weight_decay"]}_epochs={epochs}_models'+\
                                f'/model_{epoch}'
                        else:
                            model_save_dir = f'./trained_models_{params.dataset}/{self.kwargs["weight_args"]["weighting"]}_{model_type}_{self.optim_param["optim"]}_lr={self.optim_param["lr"]}_bs={params.bs}_wd={self.optim_param["weight_decay"]}_epochs={epochs}_models'+\
                                f'/model_{epoch}'
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        torch.save(self.model.encoder.state_dict(), f'{model_save_dir}/encoder')
                        torch.save(self.model.decoders.state_dict(), f'{model_save_dir}/decoders')                    
                
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
                        elif self.kwargs['weight_args']['weighting'] in ['EW_STORM', 'MoCoPlus']:
                            # sampling twice needed in first epoch, else use stored sample
                            if epoch == 0:
                                self.model.prev_batch = self._process_data(train_loader)
                            else:
                                train_inputs, train_gts = self.model.prev_batch
                            train_preds = self.model(train_inputs)
                            train_preds = self.process_preds(train_preds)
                            train_losses = self._compute_loss(train_preds, train_gts)
                            self.meter.update(train_preds, train_gts)

                            # calculate loss using second batch
                            train_inputs_, train_gts_ = self.model.prev_batch
                            train_inputs_, train_gts_ = self._process_data(train_loader)
                            train_preds_ = self.model(train_inputs_)
                            train_preds_ = self.process_preds(train_preds_)
                            train_losses_ = self._compute_loss(train_preds_, train_gts_)
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

                        elif self.kwargs['weight_args']['weighting'] in ['EW_STORM', 'MoCoPlus']:
                            # compute half batch losses and gradients at a time to be memory efficient, for office-home
                            if "Art" in self.task_name:
                                grads = 0
                                grads_ = 0
                                its = 2
                                # init prev batch collector to calcualte gradient with next iterate
                                if epoch == 0:
                                    self.model.prev_batch = {task:[] for task in self.task_name}
                                    self.decoder_grads = {task:0 for task in self.task_name}
                                for i in range(its):
                                    
                                    # calculate loss using first batch
                                    train_lossesi = torch.zeros(self.task_num).to(self.device)
                                    for tn, task in enumerate(self.task_name):
                                        # sampling twice needed in first epoch, else use stored sample
                                        if epoch == 0:
                                            train_input, train_gt = self._process_data(train_loader[task])
                                        else:
                                            train_input, train_gt = self.model.prev_batch[task][i]
                                        train_pred = self.model(train_input, task)
                                        train_pred = train_pred[task]
                                        train_pred = self.process_preds(train_pred, task)
                                        train_lossesi[tn] = self._compute_loss(train_pred, train_gt, task)
                                        self.meter.update(train_pred, train_gt, task)
                                    # calculate loss using second batch
                                    train_lossesi_ = torch.zeros(self.task_num).to(self.device)
                                    for tn, task in enumerate(self.task_name):
                                        train_input_, train_gt_ = self._process_data(train_loader[task])
                                        if epoch == 0:
                                            if i==0:
                                                self.model.prev_batch[task] = [(train_input_, train_gt_)]
                                            else:
                                                self.model.prev_batch[task].append((train_input_, train_gt_))
                                        else:
                                            self.model.prev_batch[task][i] = (train_input_, train_gt_)
                                        train_pred_ = self.model(train_input_, task)
                                        train_pred_ = train_pred_[task]
                                        train_pred_ = self.process_preds(train_pred_, task)
                                        train_lossesi_[tn] = self._compute_loss(train_pred_, train_gt_, task)
                                    
                                    # calculating gradients to take average
                                    grads += self.model._get_grads(train_lossesi, mode='backward', retain_graph=True)
                                    grads_ += self.model._get_grads(train_lossesi_, mode='backward', retain_graph=True)

                                grads = grads/its
                                grads_ = grads_/its
                                for task in self.task_name:
                                    grads_ = []
                                    for p in self.model.decoders[task].parameters():
                                        grads_.append(p.grad.data/its)
                                    self.decoder_grads[task] = grads_
                            
                            # for office-31
                            else:
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
                    if self.kwargs['weight_args']['weighting'] in ['EW_STORM', 'MoCoPlus']:
                        for p in self.optimizer.param_groups:
                            p['lr'] = self.model.alpha
                        if "Art" in self.task_name:
                            w = self.model.backward(grads, grads_, input_grads=True, **self.kwargs['weight_args'])
                            # add decoder grads manually
                            for task in self.task_name:
                                for p_idx, p in enumerate(self.model.decoders[task].parameters()):
                                    p.grad.data = self.decoder_grads[task][p_idx]
                        else:
                            w = self.model.backward(train_losses, train_losses_, **self.kwargs['weight_args'])
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

            # save final model
            if save_model:
                if self.kwargs["weight_args"]["weighting"]=="ITL":
                    model_save_dir = f'./trained_models_{params.dataset}/{self.kwargs["weight_args"]["weighting"]}_task-{self.kwargs["weight_args"]["task_idx"]}_{model_type}_{self.optim_param["optim"]}_lr={self.optim_param["lr"]}_bs={params.bs}_wd={self.optim_param["weight_decay"]}_epochs={epochs}_models'+\
                        f'/model_{epochs}'
                else:
                    model_save_dir = f'./trained_models_{params.dataset}/{self.kwargs["weight_args"]["weighting"]}_{model_type}_{self.optim_param["optim"]}_lr={self.optim_param["lr"]}_bs={params.bs}_wd={self.optim_param["weight_decay"]}_epochs={epochs}_models'+\
                        f'/model_{epochs}'
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(self.model.encoder.state_dict(), f'{model_save_dir}/encoder')
                torch.save(self.model.decoders.state_dict(), f'{model_save_dir}/decoders')     

            self.meter.display_best_result()
            if return_weight:
                return self.batch_weight
    
    epochs=100

    officeModel = Officetrainer(task_dict=task_dict, 
                          weighting=weighting_method.__dict__[params.weighting], 
                          architecture=architecture_method.__dict__[params.arch], 
                          encoder_class=Encoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)
    officeModel.train(train_dataloaders=train_dataloaders, 
                      val_dataloaders=val_dataloaders,
                      test_dataloaders=test_dataloaders, 
                      epochs=epochs, save_model=False, save_model_int=2) #ADDED last two arguments to save models
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
