from __future__ import absolute_import, division, print_function
import numpy as np
from .utils import _initial_check,_get_sigmas,_get_sums
from scipy.ndimage.filters import generic_laplace,uniform_filter,correlate
from scipy import signal

def mse (GT,P):
	"""calculates mean squared error (mse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- mse value.
	"""
	GT,P = _initial_check(GT,P)
	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)

def rmse (GT,P):
	"""calculates root mean squared error (rmse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- rmse value.
	"""
	GT,P = _initial_check(GT,P)
	return np.sqrt(mse(GT,P))

def _rmse_sw_single (GT,P,ws):	
	errors = (GT-P)**2
	errors = uniform_filter(errors,ws)
	rmse_map = np.sqrt(errors)
	s = int(np.round((ws/2)))
	return np.mean(rmse_map[s:-s,s:-s]),rmse_map

def rmse_sw (GT,P,ws=8):
	"""calculates root mean squared error (rmse) using sliding window.

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  tuple -- rmse value,rmse map.	
	"""
	GT,P = _initial_check(GT,P)

	rmse_map = np.zeros(GT.shape)
	vals = np.zeros(GT.shape[2])
	for i in range(GT.shape[2]):
		vals[i],rmse_map[:,:,i] = _rmse_sw_single (GT[:,:,i],P[:,:,i],ws) 

	return np.mean(vals),rmse_map

def psnr (GT,P,MAX=None):
	"""calculates peak signal-to-noise ratio (psnr).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- psnr value in dB.
	"""
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
	"""calculates universal image quality index (uqi).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- uqi value.
	"""
	GT,P = _initial_check(GT,P)
	return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])

def _ssim_single (GT,P,ws,C1,C2):
	GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,ws)
	sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT,P,ws)

	ssim_map = ((2*GT_P_sum_mul + C1)*(2*sigmaGT_P + C2))/((GT_sum_sq + P_sum_sq + C1)*(sigmaGT_sq + sigmaP_sq + C2))

	v1 = 2 * sigmaGT_P + C2
	v2 = sigmaGT_sq + sigmaP_sq + C2

	s = int(np.round(ws/2))
	return np.mean(ssim_map[s:-s,s:-s]), np.mean(v1 / v2)


def ssim (GT,P,ws=11,K1=0.01,K2=0.03,MAX=None):
	"""calculates structural similarity index (ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  tuple -- ssim value, cs value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	C1 = (K1*MAX)**2
	C2 = (K2*MAX)**2
	ssims = []
	css = []
	for i in range(GT.shape[2]):
		ssim,cs = _ssim_single(GT[:,:,i],P[:,:,i],ws,C1,C2)
		ssims.append(ssim)
		css.append(cs)
	return np.mean(ssims),np.mean(css)


def ergas(GT,P,h_over_l=4,ws=8):
	"""calculates erreur relative globale adimensionnelle de synthese (ergas).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param h_over_l: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 8).

	:returns:  float -- ergas value.
	"""
	GT,P = _initial_check(GT,P)

	rmse_map = None
	nb = 1

	_,rmse_map = rmse_sw(GT,P,ws)

	means_map = uniform_filter(GT,ws)/ws**2

	# Avoid division by zero
	idx = means_map == 0
	means_map[idx] = 1
	rmse_map[idx] = 0

	ergasroot = np.sqrt(np.sum(((rmse_map**2)/(means_map**2)),axis=2)/nb)
	ergas_map = 100*h_over_l*ergasroot;

	s = int(np.round(ws/2))
	return np.mean(ergas_map[s:-s,s:-s])

def _scc_single(GT,P,fltr,ws):
	def _scc_filter(input, axis, output, mode, cval):
		return correlate(input, fltr , output, mode, cval, 0)

	GT_hp = generic_laplace(GT, _scc_filter)
	P_hp = generic_laplace(P, _scc_filter)
	sigmaGT_sq,sigmaP_sq,sigmaGT_P = _get_sigmas(GT_hp,P_hp,ws)
	return sigmaGT_P /(np.sqrt(sigmaGT_sq) * np.sqrt(sigmaP_sq))

def scc(GT,P,fltr=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],ws=8):
	"""calculates spatial correlation coefficient (scc).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param fltr: high pass filter for spatial processing (default=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).
	:param ws: sliding window size (default = 8).

	:returns:  float -- scc value.
	"""
	GT,P = _initial_check(GT,P)

	coefs = np.zeros(GT.shape)
	for i in range(GT.shape[2]):
		coefs[:,:,i] = _scc_single(GT[:,:,i],P[:,:,i],fltr,ws)
	return np.mean(coefs)


def rase(GT,P,ws=8):
	"""calculates relative average spectral error (rase).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- rase value.
	"""
	GT,P = _initial_check(GT,P)

	_,rmse_map = rmse_sw(GT,P,ws)

	GT_means = uniform_filter(GT, ws)/ws**2


	N = GT.shape[2]
	M = np.sum(GT_means,axis=2)/N
	rase_map = (100./M) * np.sqrt( np.sum(rmse_map**2,axis=2) / N )

	s = int(np.round(ws/2))
	return np.mean(rase_map[s:-s,s:-s])


def sam (GT,P):
	"""calculates spectral angle mapper (sam).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- sam value.
	"""
	GT,P = _initial_check(GT,P)

	GT = GT.reshape((GT.shape[0]*GT.shape[1],GT.shape[2]))
	P = P.reshape((P.shape[0]*P.shape[1],P.shape[2]))

	N = GT.shape[1]
	sam_angles = np.zeros(N)
	for i in range(GT.shape[1]):
		val = np.clip(np.dot(GT[:,i],P[:,i]) / (np.linalg.norm(GT[:,i])*np.linalg.norm(P[:,i])),-1,1)		
		sam_angles[i] = np.arccos(val)

	return np.mean(sam_angles)

def msssim (GT,P,weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],ws=11,K1=0.01,K2=0.03,MAX=None):
	"""calculates multi-scale structural similarity index (ms-ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
	:param ws: sliding window size (default = 11).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- ms-ssim value.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	GT,P = _initial_check(GT,P)

	scales = len(weights)

	mssim = []
	mcs = []
	for _ in range(scales):
		_ssim, _cs = ssim(GT, P, ws=ws,K1=K1,K2=K2,MAX=MAX)
		mssim.append(_ssim)
		mcs.append(_cs)

	mssim = np.array(mssim)
	mcs = np.array(mcs)

	filtered = [uniform_filter(im, 2)/4 for im in [GT, P]]
	GT, P = [x[::2, ::2, :] for x in filtered]
	return (np.prod(mcs[0:scales-1] ** weights[0:scales-1]) * \
		(mssim[scales-1] ** weights[scales-1]))