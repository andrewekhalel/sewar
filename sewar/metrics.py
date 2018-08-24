from __future__ import absolute_import, division, print_function
import numpy as np
from .utils import _initial_check
from scipy.ndimage.filters import convolve

def mse (GT,P):
	_initial_check(GT,P)
	return np.mean((GT-P)**2)

def rmse (GT,P):
	_initial_check(GT,P)
	return np.sqrt(mse(GT,P))

def _rmse_sw_single (GT,P,ws):
	window = np.ones((ws,ws))/(ws**2)
	errors = (GT-P)**2
	errors = convolve(errors,window,)
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

def _uqi_single(GT,P,bs=8):
	N = bs**2
	window = np.ones((bs,bs))

	GT_sq = GT*GT
	P_sq = P*P
	GT_P = GT*P

	GT_sum = convolve(GT, window)    
	P_sum =  convolve(P, window)     
	GT_sq_sum = convolve(GT_sq, window)  
	P_sq_sum = convolve(P_sq, window)  
	GT_P_sum = convolve(GT_P, window) 

	GT_P_sum_mul = GT_sum*P_sum
	GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
	numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
	denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
	denominator = denominator1*GT_P_sum_sq_sum_mul

	q_map = np.ones(denominator.shape)
	index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
	q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
	index = (denominator != 0)
	q_map[index] = numerator[index]/denominator[index]

	s = int(np.round(bs/2))
	return np.mean(q_map[s:-s,s:-s])

def uqi (GT,P,bs=8):
	_initial_check(GT,P)

	if len(GT.shape) == 2:
		return _uqi_single(GT,P,bs)
	else:
		return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],bs) for i in range(GT.shape[2])])