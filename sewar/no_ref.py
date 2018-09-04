from __future__ import absolute_import, division, print_function
import numpy as np
from .full_ref import uqi
from scipy.ndimage.filters import uniform_filter
from scipy.misc import imresize

def d_lambda (ms,fused,p=1):
	"""calculates Spectral Distortion Index (D_lambda).

	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param p: parameter to emphasize large spectral differences (default = 1).

	:returns:  float -- D_lambda.
	"""
	L = ms.shape[2]

	M1 = np.zeros((L,L))
	M2 = np.zeros((L,L))

	for l in range(L):
		for r in range(l,L):
			M1[l,r] = M1[r,l] = uqi(fused[:,:,l],fused[:,:,r])
			M2[l,r] = M2[r,l] = uqi(ms[:,:,l],ms[:,:,r])

	diff = np.abs(M1 - M2)**p
	return (1./(L*(L-1)) * np.sum(diff))**(1./p)

def d_s (pan,ms,fused,q=1,r=4,ws=7):
	"""calculates Spatial Distortion Index (D_S).

	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).
	
	:returns:  float -- D_S.
	"""
	pan = pan.astype(np.float64)
	fused = fused.astype(np.float64)

	pan_degraded = uniform_filter(pan.astype(np.float64), size=ws)/(ws**2)
	pan_degraded = imresize(pan_degraded,(pan.shape[0]//r,pan.shape[1]//r))

	L = ms.shape[2]

	M1 = np.zeros(L)
	M2 = np.zeros(L)

	for l in range(L):
		M1[l] = uqi(fused[:,:,l],pan)
		M2[l] = uqi(ms[:,:,l],pan_degraded)

	diff = np.abs(M1 - M2)**q
	return ((1./L)*(np.sum(diff)))**(1./q)

def qnr (pan,ms,fused,alpha=1,beta=1,p=1,q=1,r=4,ws=7):
	"""calculates Quality with No Reference (QNR).

	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param alpha: emphasizes relevance of spectral distortions to the overall.
	:param beta: emphasizes relevance of spatial distortions to the overall.
	:param p: parameter to emphasize large spectral differences (default = 1).
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).
	
	:returns:  float -- QNR.
	"""
	a = (1-d_lambda(ms,fused,p=p))**alpha
	b = (1-d_s(pan,ms,fused,q=q,ws=ws,r=r))**beta
	return a*b