import numpy as np
from matplotlib import pyplot as plt
import librosa
import sounddevice as sd
from time import sleep
import datetime

freqs = np.logspace(np.log10(50), np.log10(8000), 40)
# freqs = np.logspace(np.log10(50), np.log10(8000), 3)
levels = np.arange(0.7, 0.8, 0.2)

H = np.zeros([len(freqs), len(levels)])

for ifreq, freq in enumerate(freqs): 
    for ilevel, level in enumerate(levels):
        print(f'Playing freq {freq} and level {level}')
        fs = 16000
        duration = 1
        t = np.arange(0, duration, 1/fs)
        yin = np.cos(2 * np.pi * freq * t) * level
        # sd.play(yin, fs)
        # yout = sd.rec(int(duration * fs), , channels=1)
        yout = sd.playrec(yin, samplerate=fs, channels=1)
        sd.wait()
        H[ifreq, ilevel] = np.sqrt(np.mean(yout**2))
        sleep(0.2)
        
H = np.concatenate((np.expand_dims(freqs, 1), H), 1)
np.savetxt(f'H-{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.csv', H, delimiter=',', header='freqHz, level=0.6, level=0.8')        

plt.plot(H[:, 0], H[:, 1], '--ko')
plt.show()

# read from file 
H = np.genfromtxt('H-11-30-2021-22-56-07.csv', delimiter=',')
plt.plot(H[:, 0], H[:, 1], '--ko')
plt.xlabel('frequency')
plt.ylabel('rms magnitude')
plt.title('Magnitude of cone transfer function')
plt.show()
