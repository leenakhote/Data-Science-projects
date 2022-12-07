from keras.callbacks import ModelCheckpoint
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D, GlobalMaxPooling1D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import os
import pickle
from cfg import Config as config
import sklearn, joblib


def check_data():
    if os.path.isfile(config.p_path):
        print("loading existing file ".format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else :
        return None

def build_rand_feat():
    tmp = check_data()
    if tmp :
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min , _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df1[df1.label==rand_class].index)
        x = df1.iloc[file]
        rate, wav = wavfile.read('../clean/' + str(x['name']))
        label = df1.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0] - config.step)
        sample = wav[rand_index: rand_index+ config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt= config.nflit ,nfft=config.nfft)
        _min =  min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min)/ (_max - _min)
    if config.mode == 'conv':
        X = (X.reshape(X.shape[0], X.shape[1], X.shape[2]), 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=5)
    config.data = (X, y)
    with open(config.p_path1, 'wb') as handle :
        pickle.dump(config, handle, protocol=2)
    return X , y

def build_mfcc(config):
    'builds 64 mfcc features (64x9)'
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    c = config
    X = []
    y = []
    _min, _max = float('inf'), float('-inf')
    for f in tqdm(df1.index):
        x = df1.iloc[f]
        rate, wav = wavfile.read('../clean/'+str(x['name']))
        wav = wav.astype(float)
        label = df1.at[f, 'label']
        step = c.step
        for i in range(0, len(wav), step):
            partition = i+step
            if step > wav.shape[0]:
                signal = np.zeros((step, 1))
                signal[:wav.shape[0], :] = wav.reshape(-1, 1)
                X_mfcc = mfcc(signal, rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            elif partition > len(wav):
                X_mfcc = mfcc(wav[-step:], rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            else:
                X_mfcc = mfcc(wav[i:i+step], rate,
                                   numcep=c.nfeat, nfilt=c.nfeat, nfft=c.nfft).T
            _min = min(np.amin(X_mfcc), _min)
            _max = max(np.amax(X_mfcc), _max)
            X.append(X_mfcc)
            y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    print(_min, _max)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes=5)
    # with open(config.p_path1, 'wb') as handle :
    #     pickle.dump(config, handle, protocol=2)
    return X, y

def build_1d(config):
    'builds 1d data simply to show 1d convolutions are possible'
    c = config
    X = []
    y = []
    for f in df1.index:
        x = df1.iloc[f]
        rate, wav = wavfile.read('../clean/'+str(x['name']))
        label = df1.at[f, 'label']
        step = c.step
        for i in range(0, len(wav), step):
            partition = i+step
            if step > wav.shape[0]:
                signal = np.zeros((step, 1))
                signal[:wav.shape[0], :] = wav
            elif partition > len(wav):
                signal = wav[-step:]
            else:
                signal = wav[i:i+step]
            X.append(signal)
            y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    mms = MinMaxScaler()
    X = mms.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = to_categorical(y, num_classes=5)
    return X, y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding= 'same', input_shape = input_shape))
    model.add(Conv2D(32,(3,3), activation='relu', strides=(1,1), padding='same'))
    model.add(Conv2D(64,(3,3), activation='relu', strides=(1,1), padding='same'))
    model.add(Conv2D(128,(3,3), activation='relu', strides=(1,1), padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_1d_model():
    model = Sequential()
    model.add(Conv1D(16, 9, activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(Conv1D(16, 9, activation='relu', padding='same'))
    model.add(MaxPool1D(16))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(MaxPool1D(4))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(MaxPool1D(4))
    model.add(Dropout(rate=0.1))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def get_2d_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', strides=(2, 2),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


class Config:
    def __init__(self, mode='conv', nflit=26, nfeat=13, nfft=512, rate=16000 ):
        self.mode = mode
        self.nflit = nflit
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        self.p_path1 = os.path.join('pickles11')

df = pd.read_csv('baby_audio.csv')
# print (df)
df1 = df.set_index('name')
# print(df1)

for f in df1.index:
    rate, signal = wavfile.read('../clean/'+f)
    df1.at[f, 'length'] = signal.shape[0]/rate #gives length of the signal in secs

classes = list(np.unique(df1.label))
class_dist = df1.groupby(['label'])['length'].mean()
# print(class_dist)
n_samples =  2 * int(df1['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels= class_dist.index, autopct='%1.1f%%', shadow=False,startangle=90)
ax.axis('equal')
# plt.show()
df1.reset_index(inplace=True)

config = Config(mode = 'conv')

if config.mode == 'conv':
    # X, y = build_1d(config)
    # y_flat = np.argmax(y, axis=1)
    # input_shape = (X.shape[1], 1)
    # model = get_1d_model()

    # X, y = build_rand_feat()
    # y_flat = np.argmax(y, axis=1)
    # input_shape = (X.shape[1], 1)
    # model = get_conv_model()

    X, y = build_mfcc(config)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_2d_model()

elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    # model = get_recurrent_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
print(class_weight)
model_data = model.fit(X, y, epochs= 5, batch_size = 32, shuffle=True, class_weight= class_weight)

model.save(config.model_path)


model_yaml = model.to_yaml()
with open('mfcc.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights('mfcc.h5')

# with open(config.p_path1, 'w') as handle:
joblib.dump(model_data, config.p_path1)

# def save_data(data,file_path):
#     joblib.dump(data, file_path)

# with open(config.p_path1, 'wb') as handle :
#     pickle.dump(config, handle, protocol=2)

# checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode = 'max', save_best_only = True, save_weights_only=False, period=1)
#
# model.fit(X, y, epochs=5, batch_size=32,
#           shuffle=True, validation_split=0.1,
#           callbacks=[checkpoint])
#
# print(config.nflit)
# model.save(config.model_path)

