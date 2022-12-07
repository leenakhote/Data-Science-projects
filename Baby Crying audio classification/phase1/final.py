from keras.callbacks import ModelCheckpoint
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D, GlobalMaxPooling1D
from keras.layers import Dropout, Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import os
import pickle
# from cfg import Config as config
import sklearn, joblib
import librosa

X = []
wav_path_test = "/home/tickled_media_6/audio_classification/clean/burping/7E4B9C14-F955-4BED-9B03-7F3096A6CBFF-1430540826-1.0-f-26-bu.wav"
rate, signal = wavfile.read(wav_path_test)
signal = signal.astype(float)
step = int(rate/10)
nfeat = numcep = 13
nfft= 512
_min, _max = float('inf'), float('-inf')
print('signal len, step',len(signal),step)
for loop in range(0,1):
    for i in range(0, len(signal), step):
        partition = i + step
        if step > signal.shape[0]:
            signal = np.zeros((step, 1))
            signal[:signal.shape[0], :] = signal.reshape(-1, 1)
            X_mfcc = mfcc(signal, rate,numcep=nfeat, nfilt=nfeat, nfft=nfft).T
        elif partition > len(signal):
            X_mfcc = mfcc(signal[-step:], rate,numcep=nfeat, nfilt=nfeat, nfft=nfft).T
        else:
            X_mfcc = mfcc(signal[i:i + step], rate,numcep=nfeat, nfilt=nfeat, nfft=nfft).T
        _min = min(np.amin(X_mfcc), _min)
        _max = max(np.amax(X_mfcc), _max)
        X.append(X_mfcc)

X = np.array(X)
X = (X - _min) / (_max - _min)
print(X.shape)
print(X[0].shape)
X = X.reshape(1,X.shape[1], X.shape[2],1)
model = joblib.load('/home/tickled_media_6/audio_classification/classification/modelFinal.pkl')
y_pred = model.predict(X)
print(y_pred)