"""Hypercomplex (Cayley-Dickson) products used by the Q2n index.

A pixel with N = 2**n bands is treated as an element of the 2**n-dimensional
Cayley-Dickson algebra (complex numbers for N=2, quaternions for N=4,
octonions for N=8, ...), stored as its N real coefficients.

Two interchangeable backends compute the product:

* ``python``: the recursive implementation ported from the MATLAB
  pansharpening toolbox (Vivone et al.).  Always available.
* ``pyfastcd``: the compiled kernels of the optional ``pyfastcd`` package
  (``pip install sewar[fast]``).  Used automatically when it is installed.

Both backends return the same values.  Writing ``x . y`` for the standard
Cayley-Dickson product

	(a, b)(c, d) = (ac - conj(d) b, da + b conj(c)),

which is the convention ``pyfastcd`` implements, the toolbox recursion
computes

	x * y = T(conj(x . y)),

where ``T`` is the parity automorphism ``e_i -> (-1)**popcount(i) e_i``.
``T o conj`` only changes the sign of some coordinates: coordinate ``i`` keeps
its sign when ``i == 0`` or ``popcount(i)`` is odd and is negated otherwise.
The ``pyfastcd`` backend therefore multiplies with the compiled kernel and
applies that fixed sign pattern.  Q2n uses only the norm of these products,
which the sign pattern does not change, but applying it keeps the two
backends equal coordinate by coordinate.
"""
from __future__ import absolute_import, division, print_function
from functools import lru_cache
import numpy as np

try:
	import pyfastcd
except ImportError:
	pyfastcd = None


def _onion_mult_python(o1, o2):
	"""Product of two hypercomplex numbers given as length-N coefficient vectors."""
	N = len(o1)
	if N == 1:
		return o1 * o2
	L = N // 2
	a = o1[:L];  b = np.concatenate([[o1[L]], -o1[L+1:]])
	c = o2[:L];  d = np.concatenate([[o2[L]], -o2[L+1:]])
	if N == 2:
		return np.array([a[0]*c[0] - d[0]*b[0], a[0]*d[0] + c[0]*b[0]])
	ris1 = _onion_mult_python(a, c)
	ris2 = _onion_mult_python(d, np.concatenate([[b[0]], -b[1:]]))
	ris3 = _onion_mult_python(np.concatenate([[a[0]], -a[1:]]), d)
	ris4 = _onion_mult_python(c, b)
	return np.concatenate([ris1 - ris2, ris3 + ris4])

def _onion_mult2D_python(o1, o2):
	"""Pixelwise product of two H x W x N arrays of hypercomplex numbers."""
	N3 = o1.shape[2]
	if N3 == 1:
		return o1 * o2
	L = N3 // 2
	a = o1[:,:,:L];  b = np.concatenate([o1[:,:,L:L+1], -o1[:,:,L+1:]], axis=2)
	c = o2[:,:,:L];  d = np.concatenate([o2[:,:,L:L+1], -o2[:,:,L+1:]], axis=2)
	if N3 == 2:
		return np.concatenate([a*c - d*b, a*d + c*b], axis=2)
	ris1 = _onion_mult2D_python(a, c)
	ris2 = _onion_mult2D_python(d, np.concatenate([b[:,:,:1], -b[:,:,1:]], axis=2))
	ris3 = _onion_mult2D_python(np.concatenate([a[:,:,:1], -a[:,:,1:]], axis=2), d)
	ris4 = _onion_mult2D_python(c, b)
	return np.concatenate([ris1 - ris2, ris3 + ris4], axis=2)


@lru_cache(maxsize=32)
def _fastcd_context(shape):
	"""Return the pyfastcd algebra for ``shape[-1]`` and the sign pattern of
	``T o conj`` (see the module docstring) expanded to ``shape``.

	The expanded array is cached because multiplying by it is several times
	cheaper than broadcasting a length-N vector over every pixel.
	"""
	N = shape[-1]
	sign = np.array([1.0 if i == 0 or bin(i).count("1") % 2 else -1.0 for i in range(N)])
	sign = np.broadcast_to(sign, shape).copy()
	sign.flags.writeable = False
	return pyfastcd.Algebra(dimension=N), sign

def _onion_mult_fastcd(o1, o2):
	"""Product of two hypercomplex numbers given as length-N coefficient vectors."""
	N = len(o1)
	if N == 1:
		return o1 * o2
	A, sign = _fastcd_context((N,))
	return np.asarray(A(o1) * A(o2)) * sign

def _onion_mult2D_fastcd(o1, o2):
	"""Pixelwise product of two H x W x N arrays of hypercomplex numbers."""
	if o1.shape[2] == 1:
		return o1 * o2
	A, sign = _fastcd_context(o1.shape)
	out = np.empty(o1.shape, dtype=np.float64)
	A.multiply(A.asarray(o1), A.asarray(o2), out=out)
	out *= sign
	return out


if pyfastcd is not None:
	BACKEND = 'pyfastcd'
	_onion_mult, _onion_mult2D = _onion_mult_fastcd, _onion_mult2D_fastcd
else:
	BACKEND = 'python'
	_onion_mult, _onion_mult2D = _onion_mult_python, _onion_mult2D_python
