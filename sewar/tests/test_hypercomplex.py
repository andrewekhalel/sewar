from unittest import skipUnless
from .tester import Tester
import sewar
import sewar.full_ref
from sewar import _hypercomplex as hc
import numpy as np

HAVE_FASTCD = hc.pyfastcd is not None
BACKENDS = {'python': (hc._onion_mult_python, hc._onion_mult2D_python)}
if HAVE_FASTCD:
	BACKENDS['pyfastcd'] = (hc._onion_mult_fastcd, hc._onion_mult2D_fastcd)


class TestHypercomplexBackends(Tester):
	def q2n_with(self, backend, GT, P):
		saved = sewar.full_ref._onion_mult, sewar.full_ref._onion_mult2D
		sewar.full_ref._onion_mult, sewar.full_ref._onion_mult2D = BACKENDS[backend]
		try:
			return sewar.q2n(GT, P)
		finally:
			sewar.full_ref._onion_mult, sewar.full_ref._onion_mult2D = saved

	def test_selected_backend(self):
		self.assertEqual(hc.BACKEND, 'pyfastcd' if HAVE_FASTCD else 'python')
		self.assertIs(sewar.full_ref._onion_mult, BACKENDS[hc.BACKEND][0])
		self.assertIs(sewar.full_ref._onion_mult2D, BACKENDS[hc.BACKEND][1])

	def test_complex_and_norm(self):
		rng = np.random.default_rng(0)
		for mult, mult2D in BACKENDS.values():
			# N=2 is ordinary complex multiplication
			x, y = rng.standard_normal(2), rng.standard_normal(2)
			z = complex(*x) * complex(*y)
			self.assertTrue(np.allclose(mult(x, y), [z.real, z.imag]))
			# x * conj(x) = |x|^2 for every N, and |x*y| = |x||y| up to the octonions
			for N in (1, 2, 4, 8, 16):
				x, y = rng.standard_normal(N), rng.standard_normal(N)
				conj = np.concatenate([x[:1], -x[1:]])
				expected = np.zeros(N); expected[0] = np.sum(x**2)
				self.assertTrue(np.allclose(mult(x, conj), expected))
				if N <= 8:
					self.assertAlmostEqual(np.linalg.norm(mult(x, y)),
						np.linalg.norm(x) * np.linalg.norm(y))
				# the pixelwise product is the single product applied per pixel
				X, Y = rng.standard_normal((3, 5, N)), rng.standard_normal((3, 5, N))
				Z = mult2D(X, Y)
				self.assertEqual(Z.shape, X.shape)
				self.assertTrue(np.allclose(Z[2, 4], mult(X[2, 4], Y[2, 4])))

	@skipUnless(HAVE_FASTCD, "pyfastcd is not installed")
	def test_product_parity(self):
		rng = np.random.default_rng(1)
		for N in (1, 2, 4, 8, 16, 32, 64):
			for shape in ((32, 32, N), (5, 7, N)):
				X, Y = rng.standard_normal(shape), rng.standard_normal(shape)
				self.assertTrue(np.allclose(hc._onion_mult2D_fastcd(X, Y),
					hc._onion_mult2D_python(X, Y), rtol=1e-12, atol=1e-12))
			x, y = rng.standard_normal(N), rng.standard_normal(N)
			self.assertTrue(np.allclose(hc._onion_mult_fastcd(x, y),
				hc._onion_mult_python(x, y), rtol=1e-12, atol=1e-12))

	@skipUnless(HAVE_FASTCD, "pyfastcd is not installed")
	def test_product_parity_noncontiguous_and_integer(self):
		rng = np.random.default_rng(2)
		X, Y = rng.standard_normal((6, 8, 16)), rng.standard_normal((6, 8, 16))
		X, Y = X[::2, ::2, ::2], Y[::2, ::2, ::2]
		self.assertTrue(np.allclose(hc._onion_mult2D_fastcd(X, Y), hc._onion_mult2D_python(X, Y)))
		Xi = rng.integers(-9, 9, (4, 4, 4)); Yi = rng.integers(-9, 9, (4, 4, 4))
		self.assertTrue(np.array_equal(hc._onion_mult2D_fastcd(Xi, Yi), hc._onion_mult2D_python(Xi, Yi)))
		self.assertTrue(np.array_equal(hc._onion_mult_fastcd(Xi[0, 0], Yi[0, 0]),
			hc._onion_mult_python(Xi[0, 0], Yi[0, 0])))

	@skipUnless(HAVE_FASTCD, "pyfastcd is not installed")
	def test_q2n_parity(self):
		rng = np.random.default_rng(3)
		GT8 = rng.uniform(0, 255, (96, 80, 8))
		P8 = GT8 + rng.normal(0, 20, GT8.shape)
		for GT, P in ((self.read('clr'), self.read('clr_noise')),
				(self.read('gry'), self.read('gry_noise')),
				(GT8, P8), (GT8[:, :, :5], P8[:, :, :5])):
			self.assertAlmostEqual(self.q2n_with('pyfastcd', GT, P),
				self.q2n_with('python', GT, P), places=12)
