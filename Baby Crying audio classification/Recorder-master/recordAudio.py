import sounddevice as sd
from scipy.io.wavfile import write
import os
import subprocess, sys

fs = 44100  # this is the frequency sampling; also: 4999, 64000
seconds = 5  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print("Starting: Speak now!")
sd.wait()  # Wait until recording is finished
print("finished")
write('output.wav', fs, myrecording)  # Save as WAV file
# os.startfile("output.wav")


opener ="open" if sys.platform == "darwin" else "xdg-open"
subprocess.call([opener, "output.wav"])