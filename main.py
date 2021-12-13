import sounddevice as sd
import scipy.signal as scisig
import librosa
import numpy as np
import matplotlib.pyplot as plt

duration = 0.1  # seconds
fs = 44100

DEVICE_NAME_OUT = 'USBAudio2.0'
DEVICE_NAME_IN = 'JOUNIVO JV601'

DEVICE_IN = [i for i, s in enumerate(sd.query_devices()) if DEVICE_NAME_IN in s['name']]
assert(len(DEVICE_IN)==1)
DEVICE_IN = DEVICE_IN[0]

DEVICE_OUT = [i for i, s in enumerate(sd.query_devices()) if DEVICE_NAME_OUT in s['name']]
assert(len(DEVICE_OUT)==1)
DEVICE_OUT = DEVICE_OUT[0]


# for i in ransge(50):
#     y = np.random.random([10,1])
#     plt.plot(y)
#     plt.draw()
#     plt.pause(0.0001)
#     plt.clf()


plt.ion()
ax = plt.gca()

for i in range(40):

    print('starting recording')
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print('ending recording')
    print()

    rec = np.squeeze(rec)
    a = librosa.lpc(rec, 20)
    b = 1

    f, h = scisig.freqz(b, a, fs=fs)
    h  = 20 * np.log10(abs(h))

    ax.clear()
    plt.plot(f, h, 'b')

    ipeaks, _ = scisig.find_peaks(h, distance=10)
    
    # for idx, peak in np.ndenumerate(peaks):
    for ipeak in (ipeaks):
        plt.plot( np.array([f[ipeak], f[ipeak]]), np.array([-50, h[ipeak]]) )
    
    # if not peaks.size==0:
    #     for
    #     ax.vlines(x=f[peaks], color='k', linestyle='--')
    #     # plt.plot(f[peaks], h[peaks], 'x')

    # X = np.fft.rfft(rec)
    # Xpw = np.square(np.abs(X))
    # f = np.fft.rfftfreq(rec.size)
    # ax2 = ax.twinx()
    # ax2.plot(f, Xpw)

    plt.xlim(0, 5000)
    plt.ylim(-20, 80)

    # plt.show(block=False)
    plt.draw()
    plt.pause(0.0001)
    # ax.clear()
    # plt.clf()








