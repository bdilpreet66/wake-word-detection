# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sample_rate, delta_time, n_classes, batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sample_rate
        self.dt = delta_time
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        # get total number of batches
        return int(np.floor(len(self.wav_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # get the current batch of indexes
        indexes = self.indexes[index * (self.batch_size) : (index + 1) * (self.batch_size)]
        
        # get the current inputs and their expected outputs
        wav_paths = [self.wav_paths[idx] for idx in indexes]
        
        labels = [self.labels[idx] for idx in indexes]
        
        ### create the batch from inputs and outputs ###
        
        # create the structure for your batch
        X = np.empty(shape=(self.batch_size, int(self.sr * self.dt)), dtype=np.float32)
        Y = np.empty(shape=(self.batch_size, self.n_classes), dtype=np.float32)
        
        # populate data in your batch
        for idx, (path, lbl) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[idx,:] = wav.reshape(-1) # convert to column vector
            Y[idx,:] = to_categorical(lbl, self.n_classes) # convert the text into class vector i.e. [1,0,0,0....]
            
        return X, Y
        
    def on_epoch_end(self):
        # store the indexes as an array
        self.indexes = np.arange(len(self.wav_paths))
        
        # shuffle them based on the flag
        if self.shuffle:
            np.random.shuffle(self.indexes)
