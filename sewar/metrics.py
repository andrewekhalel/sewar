from __future__ import absolute_import, division, print_function
import numpy as np
from .utils import _initial_check
from scipy.ndimage.filters import convolve,gaussian_filter,uniform_filter

def mse (GT,P):
	GT,P = _initial_check(GT,P)
	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)

def rmse (GT,P):
	GT,P = _initial_check(GT,P)
	return np.sqrt(mse(GT,P))

def _rmse_sw_single (GT,P,ws):
	errors = (GT-P)**2
	errors = uniform_filter(errors,ws)
	rmse_map = np.sqrt(errors)
	s = int(np.round((ws/2)))
	return np.mean(rmse_map[s:-s,s:-s]),rmse_map

def rmse_sw (GT,P,ws=8):
	GT,P = _initial_check(GT,P)

	if len(GT.shape) == 2:
		return _rmse_sw_single (GT,P,ws)
	else:
		rmse_map = np.zeros(GT.shape)
		vals = np.zeros(GT.shape[2])
		for i in range(GT.shape[2]):
			vals[i],rmse_map[:,:,i] = _rmse_sw_single (GT[:,:,i],P[:,:,i],ws) 

		return np.mean(vals),rmse_map

def psnr (GT,P,MAX=None):
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	mse_value = mse(GT,P)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10(MAX**2 /mse_value)

def _uqi_single(GT,P,ws):
	N = ws**2
	window = np.ones((ws,ws))

	GT_sq = GT*GT
	P_sq = P*P
	GT_P = GT*P

	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)     
	GT_sq_sum = uniform_filter(GT_sq, ws)  
	P_sq_sum = uniform_filter(P_sq, ws)  
	GT_P_sum = uniform_filter(GT_P, ws) 

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

	s = int(np.round(ws/2))
	return np.mean(q_map[s:-s,s:-s])

def uqi (GT,P,ws=8):
	GT,P = _initial_check(GT,P)

	if len(GT.shape) == 2:
		return _uqi_single(GT,P,ws)
	else:
		return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])

def _ssim_single (GT,P,ws,C1,C2):
	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)     

	GT_sum_sq = GT_sum*GT_sum
	P_sum_sq = P_sum*P_sum
	GT_P_sum_mul = GT_sum*P_sum 

	sigmaGT_sq = uniform_filter(GT*GT, ws) - GT_sum_sq
	sigmaP_sq = uniform_filter(P*P, ws) - P_sum_sq
	sigmaGT_P = uniform_filter(GT*P, ws) - GT_P_sum_mul


	ssim_map = ((2*GT_P_sum_mul + C1)*(2*sigmaGT_P + C2))/((GT_sum_sq + P_sum_sq + C1)*(sigmaGT_sq + sigmaP_sq + C2))

	s = int(np.round(ws/2))
	return np.mean(ssim_map[s:-s,s:-s])


def ssim (GT,P,ws=11,K1=0.01,K2=0.03,MAX=None):
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	C1 = (K1*MAX)**2
	C2 = (K2*MAX)**2
	if len(GT.shape) == 2:
		return _ssim_single(GT,P,ws,C1,C2)
	else:
		return np.mean([_ssim_single(GT[:,:,i],P[:,:,i],ws,C1,C2) for i in range(GT.shape[2])])


def ergas(GT,P,h_over_l=4,ws=8):
	rmse_map = None
	nb = 1

	_,rmse_map = rmse_sw(GT,P,ws)
	if len(rmse_map.shape) == 2:
		rmse_map = rmse_map[:,:,np.newaxis]

	means_map = uniform_filter(GT,ws)/ws**2

	# Avoid division by zero
	idx = means_map == 0
	means_map[idx] = 1
	rmse_map[idx] = 0

	ergasroot = np.sqrt(np.sum(((rmse_map**2)/(means_map**2)),axis=2)/nb)
	ergas_map = 100*h_over_l*ergasroot;

	s = int(np.round(ws/2))
	return np.mean(ergas_map[s:-s,s:-s])