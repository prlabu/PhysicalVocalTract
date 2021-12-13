import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

def normalize_audio(y, level=0.05):
	rms = np.sqrt(np.mean(y ** 2))
	y = y / rms * level
	y[y>1] = 1
	y[y<-1] = -1
	return y

vow = 'none' # 'none','a', 'a-tube', 'i', 
glsource_type = 'gl-75hz-nonoise' # 'white-noise', 'gl', 'gl-75hz-nonoise'
acoustic_coupler = 'cone'
fname = f'coupler-{acoustic_coupler}_source-{glsource_type}_vowel-{vow}'
npzfile = np.load(f'./data/{fname}.npz') # xt=xt, ot=ot, fs=fs
xt = npzfile['xt']
ot = npzfile['ot']
fs = npzfile['fs']


ot = np.tile(ot, 1000)
ot = ot[0:(fs*100)]  # play x seconds of audio
ot = normalize_audio(ot, level=0.15)  

# plt.figure(); t=np.arange(0, len(ot))/fs; plt.plot(t[t<0.1], ot[t<0.1]); plt.show()

xt = sd.playrec(ot, samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)









