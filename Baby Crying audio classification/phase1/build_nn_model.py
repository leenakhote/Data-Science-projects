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
import sklearn, joblib
from keras.models import model_from_yaml

def build_mfcc(config):
    'builds 64 mfcc features (64x9)'
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
        self.p_path1 = os.path.join('model.pkl')


def generate_model():
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
        X, y = build_mfcc(config)
        y_flat = np.argmax(y, axis=1)
        input_shape = (X.shape[1], X.shape[2], 1)
        model = get_2d_model()
    elif config.mode == 'time':
        X, y = build_mfcc(config)
        y_flat = np.argmax(y, axis=1)
        input_shape = (X.shape[1], X.shape[2])
        # model = get_recurrent_model()

    class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
    print(class_weight)
    model_data = model.fit(X, y, epochs= 5, batch_size = 32, shuffle=True, class_weight= class_weight)
    joblib.dump(model_data, config.p_path1)

def predict_audio():
    model = joblib.load(os.path.join('model.pkl'))
    print(model)
    y_pred = model.predict(X).reshape(5)
    # model.predict(Audio)

wav_path_test = "/Users/leenakhote/leena/audio_classification/clean/burping/7E4B9C14-F955-4BED-9B03-7F3096A6CBFF-1430540826-1.0-f-26-bu.wav"
# predict_audio()

df = pd.read_csv('baby_audio.csv')
# print (df)
df1 = df.set_index('name')
# print(df1)

for f in df1.index:
    rate, signal = wavfile.read('../clean/' + f)
    df1.at[f, 'length'] = signal.shape[0] / rate  # gives length of the signal in secs

classes = list(np.unique(df1.label))
class_dist = df1.groupby(['label'])['length'].mean()
# print(class_dist)
n_samples = 2 * int(df1['length'].sum() / 0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
# plt.show()
df1.reset_index(inplace=True)

config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_mfcc(config)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_2d_model()
elif config.mode == 'time':
    X, y = build_mfcc(config)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_2d_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
print(class_weight)
model.fit(X, y, epochs=5, batch_size=32, shuffle=True, class_weight=class_weight)
y = model.predict(X[:4])
print(y)

# joblib.dump(model_data, config.p_path1)

# model = joblib.load(os.path.join(config.p_path1))
# X = model.data[0]
# y = model.data[1]
# y_pred = model.predict(X[:4]).reshape(5)
# print (y_pred)

# model_yaml = model.to_yaml()
# with open('mfcc.yaml', 'w') as yaml_file:
#     yaml_file.write(model_yaml)
# model.save_weights('mfcc.h5')

# yaml_file = open('mfcc.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# loaded_model.load_weights('mfcc.h5')
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')
# model = loaded_model
#
# print(model)