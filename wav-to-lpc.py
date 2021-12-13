import matplotlib.pyplot as plt
import numpy as np

import librosa
import scipy.signal as scisig
import os



dir = './data/test-wav/'
for fname in os.listdir(dir):
  if fname.endswith(".wav"):
    
    indata, fs = librosa.load(f"{dir}{fname}")
    print(f"Fs={fs}")
    indata = np.squeeze(indata)
    print(indata.shape)

    ypre  = librosa.effects.preemphasis(indata)
    
    fig, ax = plt.subplots()

    # x = np.copy(indata)
    # a = librosa.lpc(x, 20)
    # b = 1        
    # f, h = scisig.freqz(b, a, worN=256, fs=fs)    
    # plt.plot(f, 20 * np.log10(abs(h)), label='unfiltered')


    x = np.copy(ypre)
    a = librosa.lpc(x, 20)
    b = 1
    N = 2**10    
    lpc_f, lpc_h = scisig.freqz(b, a, worN=N, fs=fs) 
    h = 20 * np.log10(abs(h))
        
    N = x.size
    Y = np.fft.rfft(x, n=n_pts*2-1)
    Y = 20 * np.log10(np.abs(Y))
    # Y = 2 / N * (np.abs(Y))
    freqs = np.fft.rfftfreq(n_pts*2-1, 1/fs)

    plt.plot(freqs, Y, alpha=0.7, color='0.8', label='FFT')
    plt.plot(f, h, label='pre-emph')


    ax.axis((0, 5000, -50, 70))

    ax.yaxis.grid(True)
    # ax.tick_params(bottom=False, top=False, labelbottom=False,
    #                right=False, left=False, labelleft=False)
    ax.tick_params(bottom=True, top=False, labelbottom=True,
                   right=False, left=True, labelleft=True) 

    plt.legend()

    plt.title(fname)

    plt.savefig(f"./fig/{fname[0:-4]}.png")

    # plt.show()




