from __future__ import absolute_import, division, print_function
import numpy as np
from utils import _intial_check

def MSE (GT,P):
	_intial_check(GT,P)
	return np.mean((GT-P)**2)

def RMSE (GT,P):
	_intial_check(GT,P)
	return np.sqrt(MSE(GT,P))

def PSNR (GT,P,MAX=None):
	_intial_check(GT,P)
	
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	return 10 * np.log10(MAX**2 / MSE(GT,P))

