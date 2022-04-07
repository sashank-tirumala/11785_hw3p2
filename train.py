import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score
import gc
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')

from network import *
from dataloader import *

import wandb
import argparse
def setup_dataloaders(batch_size, num_proc, root = "hw3p2_student_data/hw3p2_student_data"):
    
    train_data = LibriSamples(root, 'train')
    val_data = LibriSamples(root, 'dev')
    test_data = LibriSamplesTest(root, 'test_order.csv')

    train_loader =  (DataLoader(train_data, batch_size=batch_size,collate_fn=train_data.collate_fn,
                            shuffle=True, num_workers=num_proc))# TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, collate_fn=val_data.collate_fn, num_workers=num_proc) # TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
    test_loader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, collate_fn=test_data.collate_fn)# TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 
    return train_loader, val_loader, test_loader

def setup_network( dropout_embed = 0.15, dropout_lstm = 0.35, dropout_classification = 0.2):
    model = Network(dropout_embed , dropout_lstm , dropout_classification).to(device)
    return model

def train(_lr, _b, _e, _num_proc, _de, _dl, _dc, _root_dir, _model_dir ):
    model = setup_network(_de, _dl, _dc)
    criterion = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    decoder = CTCBeamDecoder(labels=PHONEME_MAP,log_probs_input=True) 
    train_loader, val_loader, test_loader = setup_dataloaders(_b, _num_proc, _root_dir)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, _e * len(train_loader), eta_min=1e-6, last_epoch=- 1, verbose=False)
    torch.cuda.empty_cache()

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(_e):
        # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='Train') 

        total_loss = 0
        for i, data in enumerate(train_loader,0):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  
                x, y, lx, ly = data
                x = x.cuda()
                y = y.cuda()
                y = y.transpose(0,1)
                out,h = model.forward(x,lx)
                lx = lx.cpu()
                loss = criterion(out,y,h,ly)
                del x,y,out,h
                torch.cuda.empty_cache()
            total_loss += float(loss)
            wandb.log( { 'loss_step' : float(total_loss / (i + 1)) } )
            wandb.log({'lr_step': float(optimizer.param_groups[0]['lr']) })
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.
            
            batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
            batch_bar.update() # Update tqdm bar

        torch.save(model,model_dir+"hw3p2_model_"+str(epoch)+".pkl")
        torch.save(model,model_dir+"model_latest.pkl")
        wandb.save(model_dir+"model_latest.pkl")
        wandb.log({'loss_epoch' : float(total_loss / len(train_loader)), 
        'epoch': epoch+1, 
        'lr_epoch': float(optimizer.param_groups[0]['lr']) })
        batch_bar.close() # You need this to close the tqdm bar
        print("Epoch {}/{}: , Train Loss {:.04f}, Learning Rate {:.04f}".format(
        epoch + 1,
        _e,
        float(total_loss / len(train_loader)),
        float(optimizer.param_groups[0]['lr'])))




if(__name__ == "__main__"):
    import wandb

    wandb.init(project="11785-hw3p2", entity="stirumal")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument( '-lr', '--learning-rate', type=float, default=4e-3, help = 'learning rate for the training algorithm')
    parser.add_argument( '-e', '--epochs', type=int, default=100, help = 'learning rate for the training algorithm')
    parser.add_argument( '-np', '--num-proc', type=int, default=2, help = 'num_workers in dataloader')
    parser.add_argument( '-de', '--dropout_embed', type=float, default=0.15, help = 'dropout percent in embedded layer')
    parser.add_argument( '-dl', '--dropout_lstm', type=float, default=0.35, help = 'dropout percent in lstm layer')
    parser.add_argument( '-dc', '--dropout_classification', type=float, default=0.2, help = 'dropout percent in classification layer')
    parser.add_argument( '-rd', '--root_dir', type=str, default="hw3p2_student_data/hw3p2_student_data", help = 'root data directory')
    parser.add_argument( '-md', '--model_dir', type=str, default="models/", help = 'model data directory')

    args = vars(parser.parse_args())
    wandb.config.update(args) # adds all of the arguments as config variables
    print(wandb.config)
    train(_lr = args['learning_rate'], _b = args['batch_size'], _e = args['epochs'], _num_proc = args['num_proc'], _de = args['dropout_embed']
    , _dl = args['dropout_lstm'], _dc = args['dropout_classification'], _root_dir = args['root_dir'], _model_dir = args['model_dir'])
