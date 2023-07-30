from .tester import Tester
import sewar
import numpy as np

class TestMse(Tester):
	def test_color(self):
		mse = sewar.mse(self.read('clr'),self.read('clr'))
		self.assertTrue(mse == 0.0)

	def test_gray(self):
		mse = sewar.mse(self.read('gry'),self.read('gry'))
		self.assertTrue(mse == 0.0)

	def test_color_noise(self):
		mse = sewar.mse(self.read('clr'),self.read('clr_noise'))
		print (mse)
		self.assertTrue(abs(mse - 2391.465875) < self.eps)

	def test_gray_noise(self):
		mse = sewar.mse(self.read('gry'),self.read('gry_noise'))
		self.assertTrue(abs(mse - 2025.913940) < self.eps)

	def test_color_const(self):
		mse = sewar.mse(self.read('clr'),self.read('clr_const'))
		self.assertTrue(abs(mse - 2302.953958) < self.eps)

	def test_gray_const(self):
		mse = sewar.mse(self.read('gry'),self.read('gry_const'))
		self.assertTrue(abs(mse - 2016.476768) < self.eps)

class Testpsnr(Tester):
	def test_color(self):
		psnr = sewar.psnr(self.read('clr'),self.read('clr'))
		self.assertTrue(psnr == np.inf)

	def test_gray(self):
		psnr = sewar.psnr(self.read('gry'),self.read('gry'))
		self.assertTrue(psnr == np.inf)

	def test_color_noise(self):
		psnr = sewar.psnr(self.read('clr'),self.read('clr_noise'))
		print (psnr)
		self.assertTrue(abs(psnr - 14.344162) < self.eps)

	def test_gray_noise(self):
		psnr = sewar.psnr(self.read('gry'),self.read('gry_noise'))
		self.assertTrue(abs(psnr - 15.064594) < self.eps)

	def test_color_const(self):
		psnr = sewar.psnr(self.read('clr'),self.read('clr_const'))
		self.assertTrue(abs(psnr - 14.507951) < self.eps)

	def test_gray_const(self):
		psnr = sewar.psnr(self.read('gry'),self.read('gry_const'))
		self.assertTrue(abs(psnr - 15.084871) < self.eps)

class TestSsim(Tester):
	def test_color(self):
		ssim,_ = sewar.ssim(self.read('clr'),self.read('clr'),ws=11)
		self.assertTrue(ssim == 1.)

	def test_gray(self):
		ssim,_ = sewar.ssim(self.read('gry'),self.read('gry'),ws=11)
		self.assertTrue(ssim == 1.)

	def test_color_noise(self):
		ssim,_ = sewar.ssim(self.read('clr'),self.read('clr_noise'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.168002) < self.eps)

	def test_gray_noise(self):
		ssim,_ = sewar.ssim(self.read('gry'),self.read('gry_noise'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.202271) < self.eps)

	def test_color_const(self):
		ssim,_ = sewar.ssim(self.read('clr'),self.read('clr_const'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.895863) < self.eps)

	def test_gray_const(self):
		ssim,_ = sewar.ssim(self.read('gry'),self.read('gry_const'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.926078) < self.eps)


class TestRmse(Tester):
	def test_color(self):
		rmse = sewar.rmse(self.read('clr'),self.read('clr'))
		self.assertTrue(rmse == 0)

	def test_gray(self):
		rmse = sewar.rmse(self.read('gry'),self.read('gry'))
		self.assertTrue(rmse == 0)

	def test_color_sw(self):
		rmse,_ = sewar.rmse_sw(self.read('clr'),self.read('clr'))
		self.assertTrue(rmse == 0)

	def test_gray_sw(self):
		rmse,_ = sewar.rmse_sw(self.read('gry'),self.read('gry'))
		self.assertTrue(rmse == 0)

	def test_compare_rmse(self):
		rmse = sewar.rmse(self.read('gry'),self.read('gry_const'))
		rmse_sw,_ = sewar.rmse_sw(self.read('gry'),self.read('gry_const'),ws=510)
		print(rmse,rmse_sw)
		self.assertTrue(abs(rmse - rmse_sw) < self.eps)

class TestErgas(Tester):
	def test_color(self):
		ergas = sewar.ergas(self.read('clr'),self.read('clr'))
		self.assertTrue(ergas == 0)

	def test_color_noise(self):
		ergas = sewar.ergas(self.read('clr'),self.read('clr_noise'))
		self.assertTrue(abs(ergas - 10.6068) < self.eps)

	def test_gray(self):
		ergas = sewar.ergas(self.read('gry'),self.read('gry'))
		self.assertTrue(ergas == 0)

class TestScc(Tester):
	def test_color(self):
		scc = sewar.scc(self.read('clr'),self.read('clr'))
		self.assertTrue(scc == 1)

	def test_gray(self):
		scc = sewar.scc(self.read('gry'),self.read('gry'))
		self.assertTrue(scc == 1)

class TestRase(Tester):
	def test_color(self):
		rase = sewar.rase(self.read('clr'),self.read('clr'))
		self.assertTrue(rase == 0)

	def test_gray(self):
		rase = sewar.rase(self.read('gry'),self.read('gry'))
		self.assertTrue(rase == 0)

class TestSam(Tester):
	def test_color(self):
		sam = sewar.sam(self.read('clr'),self.read('clr'))
		self.assertTrue(sam < self.eps)

	def test_gray(self):
		sam = sewar.sam(self.read('gry'),self.read('gry'))
		self.assertTrue(sam < self.eps)

class TestMsssim(Tester):
	def test_color(self):
		msssim = sewar.msssim(self.read('clr'),self.read('clr'))
		self.assertTrue(msssim == 1)

	def test_gray(self):
		msssim = sewar.msssim(self.read('gry'),self.read('gry'))
		self.assertTrue(msssim == 1)

	def test_against_matlab(self):
		msssim = sewar.msssim(self.read('gry'),self.read('gry_noise'))
		print (msssim)
		self.assertTrue(abs(0.631429952770791-msssim)<self.eps)

class TestNoRef(Tester):
	def test_color(self):
		d = sewar.no_ref.d_lambda(self.read('clr'),self.read('clr'))
		self.assertTrue(d == 0)

class TestVIF(Tester):
	def test_color(self):
		v = sewar.full_ref.vifp(self.read('clr'),self.read('clr'))
		print(v)
		self.assertTrue((1-v)<self.eps )

	def test_gray(self):
		v = sewar.full_ref.vifp(self.read('gry'),self.read('gry'))
		self.assertTrue((1-v)<self.eps)

	def test_gray_noise(self):
		v = sewar.full_ref.vifp(self.read('gry'),self.read('gry_noise'))
		print (v)
		self.assertTrue(abs(0.120490551257006-v)<self.eps)

	def test_gray_const(self):
		v = sewar.full_ref.vifp(self.read('gry'),self.read('gry_const'))
		print (v)
		self.assertTrue(abs(0.981413452665522-v)<self.eps)


class TestPSNRB(Tester):
	def test_gray_noise(self):
		v = sewar.full_ref.psnrb(self.read('gry'),self.read('gry_noise'))
		print (v)
		self.assertTrue(abs(15.0646-v)<self.eps)

	def test_color_noise(self):
		v = sewar.full_ref.psnrb(self.read('clr'),self.read('clr_noise'))
		print (v)
		# Value computed for psnrb using MATLAB for the first channel of the image. 
		# There could be discrepancy in the results compared to MATLAB when using the Y channel of the image, due to scaling issues in Python vs MATLAB 
		self.assertTrue(abs(14.5881-v)<self.eps)
