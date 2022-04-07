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
#import ctcdecode
#import Levenshtein
#from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
PHONEME_MAP = [
    " ",
    ".", #SIL
    "a", #AA
    "A", #AE
    "h", #AH
    "o", #AO
    "w", #AW
    "y", #AY
    "b", #B
    "c", #CH
    "d", #D
    "D", #DH
    "e", #EH
    "r", #ER
    "E", #EY
    "f", #F
    "g", #G
    "H", #H
    "i", #IH 
    "I", #IY
    "j", #JH
    "k", #K
    "l", #L
    "m", #M
    "n", #N
    "N", #NG
    "O", #OW
    "Y", #OY
    "p", #P 
    "R", #R
    "s", #S
    "S", #SH
    "t", #T
    "T", #TH
    "u", #UH
    "U", #UW
    "v", #V
    "W", #W
    "?", #Y
    "z", #Z
    "Z" #ZH
]
import csv
class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= "train"): # You can use partition to specify train or dev

        self.X_dir = data_path + "/" + partition + "/mfcc/"# TODO: get mfcc directory path
        self.Y_dir = data_path + "/" + partition +"/transcript/"# TODO: get transcript path

        self.X_files = os.listdir(self.X_dir) # TODO: list files in the mfcc directory
        self.Y_files = os.listdir(self.Y_dir)# TODO: list files in the transcript directory

        # TODO: store PHONEMES from phonemes.py inside the class. phonemes.py will be downloaded from kaggle.
        # You may wish to store PHONEMES as a class attribute or a global variable as well.
        self.PHONEMES = ["", 'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH']

        assert(len(self.X_files) == len(self.Y_files))

        pass

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):

        X_path = self.X_dir + self.X_files[ind]
        Y_path = self.Y_dir + self.Y_files[ind]
    
        X = torch.tensor(np.load(X_path))# TODO: Load the mfcc npy file at the specified index ind in the directory
        Y = np.load(Y_path)# TODO: Load the corresponding transcripts
        Y1 = Y[1:-1]
        # Remember, the transcripts are a sequence of phonemes. Eg. np.array(['<sos>', 'B', 'IH', 'K', 'SH', 'AA', '<eos>'])
        # You need to convert these into a sequence of Long tensors
        # Tip: You may need to use self.PHONEMES
        # Remember, PHONEMES or PHONEME_MAP do not have '<sos>' or '<eos>' but the transcripts have them. 
        # You need to remove '<sos>' and '<eos>' from the trancripts. 
        # Inefficient way is to use a for loop for this. Efficient way is to think that '<sos>' occurs at the start and '<eos>' occurs at the end.
        
        Yy = torch.tensor([self.PHONEMES.index(yy) for yy in Y1]) # TODO: Convert sequence of  phonemes into sequence of Long tensors

        return X, Yy
    
    def collate_fn(self,batch):

        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x]# TODO: Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [len(y) for y in batch_y]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


#You can either try to combine test data in the previous class or write a new Dataset class for test data
import pdb
class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order): # test_order is the csv similar to what you used in hw1
        with open(data_path + '/test/'+test_order, newline='') as f:
          reader = csv.reader(f)
          test_order_list = list(reader)
         # TODO: open test_order.csv as a list
        #pdb.set_trace()
        self.X = [torch.tensor(np.load(data_path + '/test/mfcc/' + X_path[0])) for X_path in test_order_list[1:]] # TODO: Load the npy files from test_order.csv and append into a list
        # You can load the files here or save the paths here and load inside __getitem__ like the previous class
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        # TODOs: Need to return only X because this is the test dataset
        return self.X[ind]
    
    def collate_fn(self,batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)

if (__name__ == "__main__"):
    batch_size = 128
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

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))
    for data in val_loader:
        x, y, lx, ly = data # if you face an error saying "Cannot unpack", then you are not passing the collate_fn argument
        print(x.shape, y.shape, lx.shape, ly.shape)
        break
    pass