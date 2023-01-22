import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from audio_mae import AudioMaskedAutoencoderViT

from functools import partial
from tqdm import tqdm

import os
from os.path import join as pj
import yaml
from yaml import Loader
import numpy as np

from dataset import Dataset
from utils import *


def main(args):
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=Loader)
        
    init_project(config)

    train_dataset = Dataset(config['data']['train'],
                        sr=config['data']['params']['sr'], 
                        n_mels=config['data']['params']['n_mels'], 
                        n_fft=config['data']['params']['n_fft'], 
                        hop_length=config['data']['params']['hop_length'], 
                        win_length=config['data']['params']['win_length'], 
                        ms=config['data']['params']['ms'],
                        augs=True)

    test_dataset = Dataset(config['data']['train'],
                        sr=config['data']['params']['sr'], 
                        n_mels=config['data']['params']['n_mels'], 
                        n_fft=config['data']['params']['n_fft'], 
                        hop_length=config['data']['params']['hop_length'], 
                        win_length=config['data']['params']['win_length'], 
                        ms=config['data']['params']['ms'],
                        augs=False)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['n_workers'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['n_workers'])

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

#     audio_mels = torch.ones([2, 1, 1024, 128])
    
    # Paper recommended archs
    model  = AudioMaskedAutoencoderViT(
        num_mels=config['model']['num_mels'], 
        mel_len=config['model']['mel_len'], 
        in_chans=config['model']['in_chans'],
        patch_size=config['model']['patch_size'], 
        embed_dim=config['model']['embed_dim'], 
        encoder_depth=config['model']['encoder_depth'], 
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'], 
        decoder_depth=config['model']['decoder_depth'], 
        decoder_num_heads=config['model']['decoder_num_heads'],
        mlp_ratio=config['model']['mlp_ratio'], 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['training']['factor'], \
         patience=config['training']['patience'], verbose=True, min_lr=config['training']['min_lr'])

    best_loss = float('inf')
 
    warmup_steps = config['training']['warmup']
    lr = config['training']['lr']

    for epoch in range(config['training']['epochs']):
        print(f'Epoch: {epoch}')
        model.train()
        model, train_loss = train(model, optimizer, warmup_steps, lr, scheduler, train_dataloader, epoch)

        model.eval()
        test_loss = test(model, test_dataloader)

        metrics = {"Test loss": test_loss}

        wandb_log_metrics(metrics)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model, pj('runs', config['exp_label'], 'weights', 'pth', 'best_loss.pth'))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    





