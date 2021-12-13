from matplotlib.mlab import psd, csd
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import interp
from scipy.linalg.misc import norm
import sounddevice as sd
import glottal
import scipy.signal as scisig
from scipy import interpolate
import soundfile as sf
import importlib
import librosa



def butter_bandstop_filter(data, fs, lowcut=20, highcut=30, order=2):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	i, u = scisig.butter(order, [low, high], btype='bandstop')
	y = scisig.filtfilt(i, u, data)
	return y


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
	"""
	Input :
	s: 1d-array, data signal from which to extract high and low envelopes
	dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
	split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
	Output :
	lmin,lmax : high/low envelope idx of input signal s
	"""
	# locals min
	lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
	# locals max
	lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1
	if split:
		# s_mid is zero if s centered around x-axis or more generally mean of signal
		s_mid = np.mean(s)
		# pre-sorting of locals min based on relative position with respect to s_mid
		lmin = lmin[s[lmin] < s_mid]
		# pre-sorting of local max based on relative position with respect to s_mid
		lmax = lmax[s[lmax] > s_mid]
	# global max of dmax-chunks of locals max
	lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]])
							 for i in range(0, len(lmin), dmin)]]
	# global min of dmin-chunks of locals min
	lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]])
							 for i in range(0, len(lmax), dmax)]]
	return lmin, lmax


def plotSpectrum(y, fs, n=1024, fig=None, show=True):
	"""
	Plots a Single-Sided Amplitude Spectrum of y(t)
	"""
	# N = len(y) # length of the signal
	if fig:
		plt.figure(fig.number)
	else: 
		fig, ax = plt.subplots()
	Y = np.fft.rfft(y, n=n, norm='ortho')  # fft computing and normalization
	Y = 20*np.log10(np.abs(Y))
	f = np.fft.rfftfreq(n, 1/fs)
	plt.plot(f, Y, color='0.7', alpha=0.5, label='spectrum')  # plotting the spectrum
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('|Y| dB')
	if show:
		plt.show()
	return fig, f, Y


def tfe(x, y, fs, *args, **kwargs):
	"""estimate transfer function from x to y, see csd for calling convention"""
	c, freqs = csd(y, x, Fs=fs)
	p, _ = psd(x)
	return [np.true_divide(c, p), freqs]


def fftnoise(f):
	f = np.array(f, dtype='complex')
	Np = (len(f) - 1) // 2
	phases = np.random.rand(Np) * 2 * np.pi
	phases = np.cos(phases) + 1j * np.sin(phases)
	f[1:Np+1] *= phases
	f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
	return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
	freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
	f = np.zeros(samples)
	idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
	f[idx] = 1
	return fftnoise(f)


def normalize_audio(y, level=0.05):
	rms = np.sqrt(np.mean(y ** 2))
	y = y / rms * level
	y[y>1] = 1
	y[y<-1] = -1
	return y

def dB(y):
	return 20*np.log10(np.abs(y))

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return abs(x), np.angle(x)

fs = 22050
duration = 10
N = 2**16

gt = band_limited_noise(0, fs/2, fs*duration, fs)
gt = normalize_audio(gt, level=0.05)
Gw = np.fft.rfft(gt, n=N, norm='ortho')
Gw_f = np.fft.rfftfreq(N, 1/fs)

fbase = 'white-noise-cone-5min'  # 'white-noise' or 'zero'
# yin, fs = sf.read(f'./data/{fbase}-in.wav')
ot, fs = sf.read(f'./data/{fbase}-out.wav')

# ot = sd.playrec(gt, samplerate=fs, channels=1); sd.wait(); ot = np.squeeze(ot)
# sf.write('./data/white-noise-cone-5min-out.wav', ot, fs)

N = len(ot)
Sw = np.fft.rfft(ot, n=N, norm='ortho')  # fft computing and normalization
f = np.fft.rfftfreq(N, 1/fs)

fig, axs = plt.subplots(2, 1); plt.ion()
# fig = plotSpectrum(ot, fs, len(ot), fig=fig, show=False)

ax = axs[0]
ax.plot(f, 20*np.log10(np.abs(Sw)), color='0.8', label='spectrum')  # plotting the spectrum
ax.set_xlabel('')
ax.set_ylabel('|Y| (dB)')

ax = axs[1]
ax.plot(f, (np.angle(Sw)), color='0.8', label='spectrum')  # plotting the spectrum
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('∠Y (rad)')

y = np.abs(Sw)
x = np.copy(f)
high_idx, low_idx = hl_envelopes_idx(y, dmin=100,dmax=100)
x = x[low_idx]; y = y[low_idx]
Sw_interp = interpolate.interp1d(x, y)
xint = np.linspace(10, fs/2-10, 10000)
yint = Sw_interp(xint)
# plt.plot(xint, yint, label='interp')

plt.suptitle('Cone transfer function')
plt.setp(axs, xlim=(0,10000))

plt.savefig('./fig/cone-transfer-function-white-noise.png', dpi=300)

axs[0].plot(x, y, color='orange', label='envelope')
axs[0].legend()
plt.savefig('./fig/cone-transfer-function-white-noise-env.png', dpi=300)




np.savez('./data/Sw-cone-5min.npz', Sw=Sw, f=f, Swinv=Swinv, Swinv_f=Swinv_f)

npzfile = np.load('./data/Sw-cone-5min.npz')
Sw = npzfile['Sw']
f = npzfile['f']
Swinv = npzfile['Swinv']
Swinv_f = npzfile['Swinv_f']


Sw_mag, Sw_ang = R2P(Sw)  # plt.figure(); plt.hist(Sw_ang)
y = np.abs(Sw_mag)
x = np.copy(f)
high_idx, low_idx = hl_envelopes_idx(y, dmin=100,dmax=100)
x = x[low_idx]; y = y[low_idx]
Sw_interp = interpolate.interp1d(x, y)


# plt.figure(); plt.plot()


Sw_clean = 0.9*np.ones(Gw_f.shape)
frange = (10, 8000)
idxs = np.logical_and(Gw_f > frange[0], Gw_f < frange[1])
Sw_clean[idxs] = Sw_interp(Gw_f[idxs]) # c
# Sw_clean[Gw_f<20] = 0.1 
# Gw_f = Gw_f[idxs]
# Gw = Gw[idxs]

# Ow = np.true_divide(np.abs(Gw), np.abs(Sw_interp(Gw_f)))
Ow_mag = 1/(Sw_clean) # plt.figure(); plt.plot(Gw_f, np.abs(Ow))
# Ow = np.true_divide(np.abs(Gw), np.abs(Sw_interp(Gw_f)))
Ow_ang = np.random.rand(len(Ow_mag)) * 2 * np.pi - np.pi
Ow = P2R(Ow_mag, Ow_ang)
Swinv = np.copy(Ow)
Swinv_f = np.copy(Gw_f)
Ow = scisig.savgol_filter(Ow, 251, 2)  # plt.figure(); plt.plot(Gw_f, dB(Ow))
ot = np.fft.irfft(Ow, norm='ortho') # plt.figure(); plt.plot(ot)
# ot = ot[round(len(ot)*0.1):round(len(ot)*0.9)]
ot = normalize_audio(ot, level=0.05) # plt.figure(); plt.hist(ot)
ot = np.tile(ot, 1000)
ot = ot[0:(fs*10)]  # play 10 seconds of audio

gst = sd.playrec(ot, samplerate=fs, channels=1); sd.wait(); gst = np.squeeze(gst)




# xt = sd.playrec(ot, samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)
# xt = sd.playrec(np.zeros(round(len(ot)/2)), samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)
# np.savez(f'./data/{fname}', xt=xt, ot=ot, fs=fs)






##### Measure with glottal pulses
fs = 22050
F0 = 75
glo = glottal.Class_Glottal(F0=F0, sampling_rate=fs, noise_level=0)

# Load function of the speaker output system
npzfile = np.load('./data/Sw-cone-5min.npz')
Sw = npzfile['Sw']
f = npzfile['f']
Swinv = npzfile['Swinv']
Swinv_f = npzfile['Swinv_f']

gt = glo.yg # 
gt = np.tile(gt, 1000) # plt.figure(); plt.plot(np.arange(0, 1200) / fs, gt[np.arange(0, 1200)])
gt = np.diff(gt, prepend=0)
gt = normalize_audio(gt, level=0.05) # xt = sd.playrec(gt, samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)
N = len(gt)
Gw = np.fft.rfft(gt, n=N, norm='ortho')
Gw_f = np.fft.rfftfreq(N, 1/fs)

Swinv_interp = interpolate.interp1d(Swinv_f, Swinv)

Ow = np.multiply(Gw, Swinv_interp(Gw_f))  # plt.figure(); plt.plot(Gw_f, dB(Ow))
ot = np.fft.irfft(Ow)
ot = normalize_audio(ot, level=0.05)
ot = np.tile(ot, 1000)
ot = ot[0:(fs*10)]  # play 10 seconds of audio




vow = 'a-tube' 
glsource_type = 'gl-75hz-nonoise' # 'white-noise', 'gl'
acoustic_coupler = 'cone'
fname = f'coupler-{acoustic_coupler}_source-{glsource_type}_vowel-{vow}'
# npzfile = np.load(f'./data/{fname}.npz') # xt=xt, ot=ot, fs=fs
# xt = npzfile['xt']
# ot = npzfile['ot']
# fs = npzfile['fs']

xt = sd.playrec(ot, samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)
# xt = sd.playrec(np.zeros(round(len(ot)/2)), samplerate=fs, channels=1); sd.wait(); xt = np.squeeze(xt)
np.savez(f'./data/{fname}', xt=xt, ot=ot, fs=fs)
###################




# plot
vow = 'i' # 'none','a', 'a-tube', 'i', 
glsource_type = 'gl-75hz-nonoise' # 'white-noise', 'gl', 'gl-75hz-nonoise'
acoustic_coupler = 'cone'
fname = f'coupler-{acoustic_coupler}_source-{glsource_type}_vowel-{vow}'
npzfile = np.load(f'./data/{fname}.npz') # xt=xt, ot=ot, fs=fs
xt = npzfile['xt']
ot = npzfile['ot']
fs = npzfile['fs']


fig, ax = plt.subplots(); plt.ion()
_, x, y = plotSpectrum(xt, fs, len(xt), fig=fig)

high_idx, low_idx = hl_envelopes_idx(y, dmin=200,dmax=200)
x = x[low_idx]; y = y[low_idx]
plt.plot(x, y, 'k', label='spec. envelope')
	
a = librosa.lpc(xt, 40)
b = 1
N = 2**12   
lpc_f, lpc_h = scisig.freqz(b, a, worN=N, fs=fs) 
lpc_h = dB(lpc_h)-10
ipks, _ = scisig.find_peaks(lpc_h); ipks = ipks[0:4]
# for ipk in ipks:
# 	plt.axvline(x=lpc_f[ipk], linestyle='--', c='0.5')
plt.vlines(x=lpc_f[ipks], ymin=-80, ymax=lpc_h[ipks], linestyle='--', color='0.5', label='F1-F4', zorder=0)
plt.plot(lpc_f, lpc_h, color='orange', label='LPC')

plt.xlim((0,5000))
plt.ylim((-80,40))
plt.title(f'Coupler: {acoustic_coupler}, Source: {glsource_type}, Filter: {vow}')
plt.legend()
plt.show()

# plotSpectrum(ot, fs, len(ot), fig=fig)
plt.savefig(f'./fig/{fname}', dpi=300)
plt.pause(1)
plt.close(fig)









gst = sd.playrec(ot, samplerate=fs, channels=1); sd.wait(); gst = np.squeeze(gst)
gst = butter_bandstop_filter(np.squeeze(gst), fs)


# rms = np.sqrt(np.mean(yin ** 2))
# yin = yin / rms * 0.05
# # yin = np.zeros(yin.shape)
# gst = sd.playrec(yin, samplerate=fs, channels=1)
# sd.wait()

# # sf.write('./data/zero-in.wav', yin, fs)
# # sf.write('./data/zero-out.wav', yout, fs)


# # b, a = scisig.iirnotch(w0=24.5, Q=30, fs=fs)
# # yout = scisig.filtfilt(b, a, yout)

# # yout = yin
# # yout = np.squeeze(yout)


# b, a = scisig.butter(3, 0.5, 'low', analog=False)
# H = scisig.filtfilt(b, a, H)
# Hmag = 20 * np.log10(np.abs(H))
# # Y = 2 / N * (np.abs(Y))
# freqs = np.fft.rfftfreq(N, 1/fs)
# plt.plot(freqs, Hmag, color='0.5', alpha=0.5)
# # Ysm = scisig.savgol_filter(Hmag, 1000, 5)
# # plt.plot(freqs, Hmag, 'k')
# plt.show()

# with open('./data/tf.npy', 'wb') as f:
#     np.save(f, H)

#     # X = np.fft.rfft(rec)
#     # Xpw = np.square(np.abs(X))
#     # f = np.fft.rfftfreq(rec.size)
# plt.figure()
# Hcone, freqs = tfe(yin, yout, fs=fs)
# plt.plot(freqs, np.abs(Hcone))
# plt.show()


# glo = glottal.Class_Glottal(sampling_rate=fs)
# x = glo.make_N_repeat(repeat_num=10000)
# X = np.fft.rfft(x)

# Xpre = np.multiply(X, Hcone)
# xpre = np.fft.ifft(Xpre)

# x = sd.playrec(xpre, glo.sr, channels=1)
# sd.wait()

x = ot_bs
nplots = 1
plt.subplot(nplots, 1, 4)
dur = 50  # in ms
istart = 4000
idxs = np.arange(istart, istart+round(dur*fs/1000))
plt.plot((np.arange(len(x[idxs])) * 1000.0 / fs), x[idxs], 'k')
ax = plt.gca()
ax.set_ylabel("in", fontsize=14, color='k')
ax.set_xlabel('time (ms)')
plt.show()


# ax2 = ax.twinx()
# plt.plot((np.arange(len(x[idxs])) * 1000.0 / glo.sr), x[idxs], 'b')
# ax2.set_ylabel("out",fontsize=14,color='b')
