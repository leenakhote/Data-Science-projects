import pandas as pd
import os
import librosa
import librosa.display

from wavefilehelper import WavFileHelper

wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath('/datasets'), 'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
# audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])