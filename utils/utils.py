import os
from os.path import join as pj
import numpy as np
from tqdm import tqdm
import wandb
import argparse

from utils.wandb_utils import init_wandb_project

def init_project(config):
    exp_label = config['exp_label']

    mkdir(pj('runs', exp_label, 'weights', 'pth'))
    mkdir(pj('runs', exp_label, 'weights', 'onnx'))

    wandb_project = config['wandb']['project']
    wandb_user = config['wandb']['user']

    init_wandb_project(exp_label, wandb_project, wandb_user)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

def adjust_optim(step, optimizer, scheduler, loss, warmup_steps, start_lr, epoch, epoch_length):
    warmup_shift = start_lr / warmup_steps
    # step = epoch * epoch_length + step
    if (step <= warmup_steps) and epoch==0: #(optimizer.param_groups[0]['lr'] < start_lr)
        optimizer.param_groups[0]['lr'] += warmup_shift
    else:  
        scheduler.step(loss)

def test(model, dataloader):
    losses = []

    for i, batch in enumerate(tqdm(dataloader)):
        tensors = batch.cuda()
        loss, pred, mask = model(tensors)

        losses.append(loss.item())
    
    return mean(losses)


def train(model, optimizer, warmup_steps, lr, scheduler, dataloader, epoch):
    losses = []
    for i, batch in enumerate(tqdm(dataloader)):
        tensors = batch.cuda()
        loss, pred, mask = model(tensors)

        optimizer.zero_grad()
        
        wandb.log({"loss": loss.item(),
                   "lr":optimizer.param_groups[0]['lr']})

        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        adjust_optim(i, optimizer, scheduler, loss.item(), warmup_steps, lr, epoch, len(dataloader))

    return model, mean(losses)

def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def mean(x):
    return sum(x)/len(x) if len(x) else 0

