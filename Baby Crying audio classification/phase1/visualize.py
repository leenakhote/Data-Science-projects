import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt

audio_obj, sampling_rate = librosa.core.load('baby_crying.wav')
print('sampling_rate',sampling_rate)
C_transform = librosa.cqt(audio_obj,sampling_rate)
print(C_transform)
amp = librosa.core.amplitude_to_db(np.abs(C_transform**2))
display.specshow(amp, x_axis='time',y_axis='cqt_note')
plt.colorbar()
plt.show()