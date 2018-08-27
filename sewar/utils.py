import numpy as np
from scipy.ndimage.filters import uniform_filter
def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes"
	assert GT.dtype == P.dtype, "Supplied images have different dtypes"

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT.astype(np.float64),P.astype(np.float64)

def _replace_value(array,value,replace_with):
    array[array == value] = replace_with
    return array

def _get_sums(GT,P,ws):
	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)     

	GT_sum_sq = GT_sum*GT_sum
	P_sum_sq = P_sum*P_sum
	GT_P_sum_mul = GT_sum*P_sum 
	return GT_sum_sq,P_sum_sq,GT_P_sum_mul

def _get_sigmas(GT,P,ws):
	GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,ws)

	sigmaGT_sq = uniform_filter(GT*GT, ws) - GT_sum_sq
	sigmaP_sq = uniform_filter(P*P, ws) - P_sum_sq
	sigmaGT_P = uniform_filter(GT*P, ws) - GT_P_sum_mul
	return sigmaGT_sq,sigmaP_sq,sigmaGT_P

