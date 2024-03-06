#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:15:21 2024

@author: dilpreet
"""

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os


def LSTM(N_CLASSES=10, SR=16000, DT=1.0):
    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                     n_mels=128,
                                     pad_end=True,
                                     n_fft=512,
                                     win_length=400,
                                     hop_length=160,
                                     sample_rate=SR,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_last',
                                     name='2d_convolution')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
    s = TimeDistributed(layers.Dense(64, activation='tanh'),
                        name='td_dense_tanh')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                             name='bidirectional_lstm')(s)
    x = layers.concatenate([s, x], axis=2, name='skip_connection')
    x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
    x = layers.MaxPooling1D(name='max_pool_1d')(x)
    x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(32, activation='relu',
                         activity_regularizer=l2(0.001),
                         name='dense_3_relu')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model



# Understanding the model
# from tensorflow.keras import layers
# from tensorflow.keras.layers import TimeDistributed, LayerNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2
# import kapre
# from kapre.composed import get_melspectrogram_layer
# import tensorflow as tf
# import os
# from scipy.io import wavfile
# import librosa
# import numpy as np

# model = Conv1D()
# layer_outputs = {}
# rate, wav = wavfile.read('clean/bed/2_0.wav')
# current_output = librosa.feature.melspectrogram(y=wav.astype(float), sr=rate, n_fft=512, hop_length=160, n_mels=128)[np.newaxis, :, :]  # Initial input is the audio data
# for layer in model.layers:
#     current_output = layer(current_output)  # Pass through the layer
#     if layer.name in ['batch_norm']:
#         current_output = tf.reshape(current_output, shape=(1, 101, 128, 1)).numpy()
#     else:
#         current_output = current_output.numpy()
#     layer_outputs[layer.name] = current_output

