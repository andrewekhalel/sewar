from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import signal
from utils import _intial_check

def MSE (GT,P):
	_intial_check(GT,P)
	return np.mean((GT-P)**2)

def RMSE (GT,P):
	_intial_check(GT,P)
	return np.sqrt(MSE(GT,P))

def RMSE_SW (GT,P,ws=8):
	_intial_check(GT,P)
	window = np.ones((ws,ws))/(ws**2)
	errors = (GT-P)**2
	errors = signal.convolve2d(errors,window)
	rmses = np.sqrt(errors)
	s = int(np.round((ws/2)))
	return np.mean(rmses[s:-s,s:-s])

def PSNR (GT,P,MAX=None):
	_intial_check(GT,P)
	
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	return 10 * np.log10(MAX**2 / MSE(GT,P))

