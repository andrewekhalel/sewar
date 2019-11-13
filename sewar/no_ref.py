from __future__ import absolute_import, division, print_function
import numpy as np
from math import log2, log10
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


def compute_bef(im, block_size=8):
	"""Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image

	C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
		vol. 20, no. 1, pp. 88-98, Jan. 2011.

	:param im: input image (numpy ndarray)
	:param block_size: Size of the block over which DCT was performed during compression
	:return: float -- bef.
	"""
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, i] - im[:, i+1]
		d_b += np.sum(np.square(diff))

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, i] - im[:, i+1]
		d_bc += np.sum(np.square(diff))

	# v_b for loop
	for j in list(v_b):
		diff = im[j, :] - im[j+1, :]
		d_b += np.sum(np.square(diff))

	# V_bc for loop
	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += np.sum(np.square(diff))

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	if d_b > d_bc:
		t = log2(block_size)/log2(min(height, width))
	else:
		t = 0

	# BEF
	bef = t*(d_b - d_bc)

	return bef


def psnrb(GT, P):
	"""Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)

	:param GT: first (original) input image in YCbCr format or Grayscale.
	:param P: second (corrected) input image in YCbCr format or Grayscale..
	:return: float -- psnr_b.
	"""
	if len(GT.shape) == 3:
		GT = GT[:, :, 0]

	if len(P.shape) == 3:
		P = P[:, :, 0]

	imdff = np.double(GT) - np.double(P)

	mse = np.mean(np.square(imdff.flatten()))
	bef = compute_bef(P)
	mse_b = mse + bef

	if np.amax(P) > 2:
		psnr_b = 10 * log10(255**2/mse_b)
	else:
		psnr_b = 10 * log10(1/mse_b)

	return psnr_b
