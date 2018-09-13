import numpy as np
from scipy.ndimage.filters import uniform_filter,gaussian_filter
import warnings
from enum import Enum

class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1

def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes " + \
	str(GT.shape) + " and " + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT.astype(np.float64),P.astype(np.float64)

def _replace_value(array,value,replace_with):
    array[array == value] = replace_with
    return array

def _get_sums(GT,P,fltr:Filter,**kwargs):
	if fltr == Filter.UNIFORM:
		GT_sum = uniform_filter(GT, size=kwargs['ws'])
		P_sum =  uniform_filter(P, size=kwargs['ws']) 
	elif fltr == Filter.GAUSSIAN:
		GT_sum = gaussian_filter(GT, sigma=kwargs['s'], truncate=kwargs['t'])
		P_sum =  gaussian_filter(P, sigma=kwargs['s'], truncate=kwargs['t']) 

	GT_sum_sq = GT_sum*GT_sum
	P_sum_sq = P_sum*P_sum
	GT_P_sum_mul = GT_sum*P_sum 
	return GT_sum_sq,P_sum_sq,GT_P_sum_mul

def _get_sigmas(GT,P,fltr:Filter,**kwargs):
	GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,fltr,**kwargs)
	if fltr == Filter.UNIFORM:
		sigmaGT_sq = uniform_filter(GT*GT, size=kwargs['ws']) - GT_sum_sq
		sigmaP_sq = uniform_filter(P*P, size=kwargs['ws']) - P_sum_sq
		sigmaGT_P = uniform_filter(GT*P, size=kwargs['ws']) - GT_P_sum_mul
	elif fltr == Filter.GAUSSIAN:
		sigmaGT_sq = gaussian_filter(GT*GT, sigma=kwargs['s'], truncate=kwargs['t']) - GT_sum_sq
		sigmaP_sq = gaussian_filter(P*P, sigma=kwargs['s'], truncate=kwargs['t']) - P_sum_sq
		sigmaGT_P = gaussian_filter(GT*P, sigma=kwargs['s'], truncate=kwargs['t']) - GT_P_sum_mul
	return sigmaGT_sq,sigmaP_sq,sigmaGT_P

def _str_to_array(str):
	pattern = r'''# Match (mandatory) whitespace between...
			(?<=\]) # ] and
			\s+
			(?= \[) # [, or
			|
			(?<=[^\[\]\s]) 
			\s+
			(?= [^\[\]\s]) # two non-bracket non-whitespace characters
			'''
	return np.array(ast.literal_eval(str))