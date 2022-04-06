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

import warnings
from dataloader import *
class Network(nn.Module):

    def __init__(self, dropout_embed = 0.15, dropout_lstm = 0.35, dropout_classification = 0.2): # You can add any extra arguments as you wish

        super(Network, self).__init__()

        # Embedding layer converts the raw input into features which may (or may not) help the LSTM to learn better 
        # For the very low cut-off you dont require an embedding layer. You can pass the input directly to the  LSTM
        self.embedding = nn.Sequential(nn.Conv1d(in_channels=13, out_channels=128, kernel_size=5, stride=2),
                                       nn.BatchNorm1d(128),nn.Dropout(dropout_embed), nn.ReLU(inplace=True),
                                       nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm1d(256), nn.Dropout(dropout_embed), nn.ReLU(inplace=True))
        
        self.lstm = nn.LSTM(256, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout= dropout_lstm ) # TODO: # Create a single layer, uni-directional LSTM with hidden_size = 256
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        self.classification = nn.Sequential(nn.Linear(512*2,2048), nn.Dropout(dropout_classification), nn.ReLU(),
                                            nn.Linear(2048, 41))# TODO: Create a single classification layer using nn.Linear()

    def forward(self, x,len_x): # TODO: You need to pass atleast 1 more parameter apart from self and x

        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        # print("x ",x.shape )
        input_for_cnn = torch.permute(x, (1,2,0))
        #print("input_for_cnn ", input_for_cnn.shape)
        input_after_cnn = self.embedding(input_for_cnn)
        #print("input_after_cnn ", input_after_cnn.shape)
        x = torch.permute(input_after_cnn, (2,0,1))
        #print("input_after_cnn ", x.shape)
        len_x = torch.clamp(len_x, max = x.shape[0])
        packed_input = pack_padded_sequence(x,len_x, enforce_sorted=False)# TODO: Pack the input with pack_padded_sequence. Look at the parameters it requires

        out1, (out2, out3) = self.lstm(packed_input)# TODO: Pass packed input to self.lstm
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths  = pad_packed_sequence(out1)# TODO: Need to 'unpack' the LSTM output using pad_packed_sequence
        #print("lengths", lengths)
        out = self.classification(out)# TODO: Pass unpacked LSTM output to the classification layer
        out_l =  F.log_softmax(out, dim=2)# Optional: Do log softmax on the output. Which dimension?
        # print("out_l", out_l.shape)
        # print("lengths", lengths.shape)
        return out_l,lengths # TODO: Need to return 2 variables

if(__name__ == "__main__"):
    model = Network().to(device)
    print(model)
    criterion = torch.nn.CTCLoss()# TODO: What loss do you need for sequence to sequence models? 
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)# TODO: Adam works well with LSTM (use lr = 2e-3)
    decoder = CTCBeamDecoder(labels=PHONEME_MAP,log_probs_input=True)
    batch_size = 16
    root = "hw3p2_student_data/hw3p2_student_data" # TODO: Where your hw3p2_student_data folder is

    train_data = LibriSamples(root, 'train')
    val_data = LibriSamples(root, 'dev')
    test_data = LibriSamplesTest(root, 'test_order.csv')
    train_loader =  (DataLoader(train_data, batch_size=batch_size,collate_fn=train_data.collate_fn,
                            shuffle=True, num_workers=2))# TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, collate_fn=val_data.collate_fn, num_workers=2) # TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
    test_loader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, collate_fn=test_data.collate_fn)# TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 
    for i, data in enumerate(train_loader, 0):
    
    # Write a test code do perform a single forward pass and also compute the Levenshtein distance
    # Make sure that you are able to get this right before going on to the actual training
    # You may encounter a lot of shape errors
    # Printing out the shapes will help in debugging
    # Keep in mind that the Loss which you will use requires the input to be in a different format and the decoder expects it in a different format
    # Make sure to read the corresponding docs about it

        x, y, lx, ly = data
        x = x.cuda()
        y = y.cuda()
        y = y.transpose(0,1)
        out,h = model.forward(x,lx)
        print(out.shape)
        print(y.shape)
        lx = lx.cpu()
        print("input_lengths", lx)
        print("logprobs", out.shape)
        print("targets", y.shape)
        print("target_lens", ly.shape)
        loss = criterion(log_probs = out,targets = y,input_lengths = h, target_lengths =ly)
        print(loss)
        del x
        torch.cuda.empty_cache()
        #out - L , B  , C
        #pdb.set_trace()

        break # one iteration is enough