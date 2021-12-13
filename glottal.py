#coding:utf-8

# glottal voice source as input of Two Tubes Model of vocal tract
# Glottal Volume Velocity 
# based on A.E.Rosenberg's formula as Glottal Volume Velocity

import numpy as np
from matplotlib import pyplot as plt

import sounddevice as sd
from scipy import signal

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1



class Class_Glottal(object):
	def __init__(self, tclosed=4.0, trise=4.5, tfall=2.0, F0=None, noise_level=0.5, sampling_rate=48000):
		# initalize
		if F0: 
			scale = (1/F0 * 1000) /  (tclosed + trise + tfall)
		else:
			scale = 1
		self.tclosed=tclosed*scale  # duration time of close state [mSec]
		self.trise=trise*scale      # duration time of opening [mSec]
		self.tfall=tfall*scale      # duration time of closing [mSec]

		self.sr= sampling_rate
		self.yg=self.make_one_plus()

		self.yg = self.yg + (np.random.rand(len(self.yg))-0.5)*noise_level
		
	def make_one_plus(self,):
		# output yg
		self.N1=int( (self.tclosed / 1000.) * self.sr )
		self.N2=int( (self.trise / 1000.) * self.sr )
		self.N3=int( (self.tfall / 1000.) * self.sr )
		self.LL= self.N1+ self.N2 + self.N3
		yg=np.zeros(self.LL)
		#print ('Length= ', self.LL)
		for t0 in range(self.LL):
			if t0 < self.N1 :
				pass
			elif t0 <= (self.N2 + self.N1):
				yg[t0]= 0.5 * ( 1.0 - np.cos( ( np.pi / self.N2 ) * (t0 - self.N1)) )
			else:
				yg[t0]= np.cos( ( np.pi / ( 2.0 * self.N3 )) * ( t0 - (self.N2 + self.N1) )  )
		return yg

	def make_N_repeat(self, repeat_num=3):
		yg_repeat=np.zeros( len(self.yg) * repeat_num)
		for loop in range( repeat_num):
			yg_repeat[len(self.yg)*loop:len(self.yg)*(loop+1)]= self.yg
		return  yg_repeat
	
	def fone(self, f):
		# calculate one point of frequecny response
		xw= 2.0 * np.pi * f / self.sr
		yi=0.0
		yb=0.0
		for v in range (0,(self.N2 + self.N3)):
			yi+=  self.yg[self.N1 + v] * np.exp(-1j * xw * v)
			yb+=  self.yg[self.N1 + v]
		val= yi/yb
		return np.sqrt(val.real ** 2 + val.imag ** 2)
	
	def H0(self, freq_low=100, freq_high=5000, Band_num=256):
		# get Log scale frequecny response, from freq_low to freq_high, Band_num points
		amp=[]
		freq=[]
		bands= np.zeros(Band_num+1)
		fcl=freq_low * 1.0    # convert to float
		fch=freq_high * 1.0   # convert to float
		delta1=np.power(fch/fcl, 1.0 / (Band_num)) # Log Scale
		bands[0]=fcl
		#print ("i,band = 0", bands[0])
		for i in range(1, Band_num+1):
			bands[i]= bands[i-1] * delta1
			#print ("i,band =", i, bands[i]) 
		for f in bands:
			amp.append(self.fone(f) )
		return   amp, bands # = amp value, freq list



if __name__ == '__main__':
	
	# instance
	fs = 22050
	glo=Class_Glottal(sampling_rate=fs)


	yp = np.diff(yg_repeat, prepend=0)
	yp_range = max(yp) - min(yp)
	yp = 2* (yp - min(yp)) / yp_range
	yp = yp - 1
	# yin = yg_repeat[1:(glo.sr*2)]
	# yin = np.cos(2*np.pi*500*np.arange(0, 5, 1/fs))
	yin = np.zeros(np.arange(0, 5, 1/fs).size)
	yout = sd.playrec(yin, fs, channels=1)
	sd.wait(); yout = np.squeeze(yout)

	# plotSpectrum(yout, fs)

	nplots = 4

	# draw
	fig = plt.figure()
	# draw one waveform
	plt.subplot(nplots,1,1)
	plt.xlabel('time (ms)')
	plt.ylabel('level')
	plt.title('Glottal waveform')
	plt.plot( (np.arange(len(glo.yg)) * 1000.0 / glo.sr) , glo.yg)

	# draw frequecny response
	plt.subplot(nplots,1,2)
	plt.xlabel('Hz')
	plt.ylabel('dB')
	plt.title('Glottal frequency response')
	amp, freq=glo.H0(freq_high=5000, Band_num=256)
	plt.plot(freq, amp)
	
	# draw repeated waveform
	yg_repeat=glo.make_N_repeat(repeat_num=3)
	yp = np.diff(yg_repeat, prepend=0)
	yp_range = max(yp) - min(yp)
	yp = 2* (yp - min(yp)) / yp_range
	yp = yp - 1
	plt.subplot(nplots,1,3)
	plt.xlabel('time (ms)')
	plt.ylabel('level')
	plt.title('Glottal repeated Waveform')
	plt.plot( (np.arange(len(yg_repeat)) * 1000.0 / glo.sr) , yg_repeat)


	plt.subplot(nplots,1,4)
	dur = 150 # in ms 
	istart = 3000
	idxs = np.arange(istart, istart+round(dur*glo.sr/1000))
	plt.plot((np.arange(len(yin[idxs])) * 1000.0 / glo.sr), yin[idxs], 'k')
	ax = plt.gca()
	ax.set_ylabel("in",fontsize=14,color='k')
	ax.set_xlabel('time (ms)')

	ax2 = ax.twinx()
	# b, a = signal.iirnotch(w0=24, Q=30, fs=fs)
	# yout = signal.filtfilt(b, a, yout)
	# b, a = signal.iirnotch(w0=50, Q=30, fs=fs)
	# yout = signal.filtfilt(b, a, yout)
	plt.plot((np.arange(len(yout[idxs])) * 1000.0 / fs), yout[idxs], 'b')
	ax2.set_ylabel("out",fontsize=14,color='b')

	fig.tight_layout()
	plt.show()

	



	
#This file uses TAB