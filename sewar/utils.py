import numpy as np
from scipy.ndimage.filters import uniform_filter,gaussian_filter
from scipy import signal
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

def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul

def fspecial(fltr,ws,**kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)

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

def _power_complex(a,b):
	return a.astype('complex') ** b