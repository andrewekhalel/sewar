from __future__ import absolute_import, division, print_function
import numpy as np
from .full_ref import uqi
from scipy.ndimage.filters import uniform_filter
from scipy.misc import imresize

def d_lambda (ms,fused,p=1):
	L = ms.shape[2]

	M1 = np.zeros((L,L))
	M2 = np.zeros((L,L))

	for l in range(L):
		for r in range(L):
			M1[l,r] = uqi(fused[:,:,l],fused[:,:,r])
			M2[l,r] = uqi(ms[:,:,l],ms[:,:,r])

	diff = np.abs(M1 - M2)**p
	return (1./(L*(L-1)) * np.sum(diff))**(1./p)

def d_s (pan,ms,fused,q=1,ws=7,r=4):
	pan_degraded = uniform_filter(pan, size=ws)/(ws**2)
	pan_degraded = imresize(pan_degraded,(pan.shape[0]//r,pan.shape[1]//r))

	L = ms.shape[2]

	M1 = np.zeros(L)
	M2 = np.zeros(L)

	for l in range(L):
		M1[l] = uqi(fused[:,:,l],pan)
		M2[l] = uqi(ms[:,:,l],pan_degraded)

	diff = np.abs(M1 - M2)**q
	return ((1./L)*(np.sum(diff)))**(1./q)

def qnr (pan,ms,fused,alpha=1,beta=1,p=1,q=1,ws=7,r=4):
	a = (1-d_lambda(ms,fused,p=p))**alpha
	b = (1-d_s(pan,ms,fused,q=q,ws=ws,r=r))**beta
	return a*b