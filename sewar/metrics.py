from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import signal
from .utils import _initial_check

def mse (GT,P):
	_initial_check(GT,P)
	return np.mean((GT-P)**2)

def rmse (GT,P):
	_initial_check(GT,P)
	return np.sqrt(mse(GT,P))

def _rmse_sw_single (GT,P,ws):
	window = np.ones((ws,ws))/(ws**2)
	errors = (GT-P)**2
	errors = signal.convolve2d(errors,window,mode="same")
	rmses = np.sqrt(errors)
	s = int(np.round((ws/2)))
	return np.mean(rmses[s:-s,s:-s])

def rmse_sw (GT,P,ws=8):
	_initial_check(GT,P)

	if len(GT.shape) == 2:
		return _rmse_sw_single (GT,P,ws)
	else:
		return np.mean([_rmse_sw_single (GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])

def psnr (GT,P,MAX=None):
	_initial_check(GT,P)
	
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	mse_value = mse(GT,P)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10(MAX**2 /mse_value)

