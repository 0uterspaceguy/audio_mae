import os
from os.path import join as pj
import torch
from torch.utils.data import Dataset as dts
import numpy as np
from random import shuffle, randint
import librosa

class Dataset(dts):
    def __init__(self,
                directory, 
                sr=16000, 
                n_mels=128, 
                n_fft=2048, 
                hop_length=256, 
                win_length=2048, 
                ms = 500,
                augs=False):
                
        self.paths = [pj(directory, audio_file) for audio_file in os.listdir(directory)]
        
        shuffle(self.paths)

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.ms = ms
        self.sample_length = int(ms/1000*sr)
        
    def preprocess(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)

        if len(y) < self.sample_length:
            y = np.pad(y, (0, self.sample_length-len(y)), 'constant', constant_values=(0, 0))
        else:
            start_point = randint(0, len(y)-self.sample_length)
            y = y[start_point:start_point+self.sample_length]
    
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=self.n_mels,
            win_length = self.win_length,
            n_fft = self.n_fft,
            hop_length = self.hop_length)
            
        mel_spec = mel_spec.T
        mel_spec = np.expand_dims(mel_spec, 0)
        return mel_spec
        
    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        tensor = self.preprocess(audio_path)
        return tensor

    def __len__(self,):
        return len(self.paths)





    

        
