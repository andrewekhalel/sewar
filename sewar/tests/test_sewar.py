from unittest import TestCase

import numpy as np
import tifffile as tif
import sewar

import os

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)
		self.RESOURCES_DIR = os.path.join(os.path.dirname(__file__),'res')
		self.IMAGES = {'clr':'lena512color.tiff',
						'clr_noise': 'lena512color_noise.tiff',
						'clr_const': 'lena512color_constant.tiff',
						'gry': 'lena512gray.tiff',
						'gry_noise': 'lena512gray_noise.tiff',
						'gry_const': 'lena512gray_constant.tiff',
						}
		self.eps = 10e-4

	def read(self,key):
		return tif.imread(os.path.join(self.RESOURCES_DIR,self.IMAGES[key]))

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
		ssim = sewar.ssim(self.read('clr'),self.read('clr'),ws=11)
		self.assertTrue(ssim == 1.)

	def test_gray(self):
		ssim = sewar.ssim(self.read('gry'),self.read('gry'),ws=11)
		self.assertTrue(ssim == 1.)

	def test_color_noise(self):
		ssim = sewar.ssim(self.read('clr'),self.read('clr_noise'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.168002) < self.eps)

	def test_gray_noise(self):
		ssim = sewar.ssim(self.read('gry'),self.read('gry_noise'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.202271) < self.eps)

	def test_color_const(self):
		ssim = sewar.ssim(self.read('clr'),self.read('clr_const'),ws=11)
		print (ssim)
		self.assertTrue(abs(ssim - 0.895863) < self.eps)

	def test_gray_const(self):
		ssim = sewar.ssim(self.read('gry'),self.read('gry_const'),ws=11)
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

	def test_gray(self):
		ergas = sewar.ergas(self.read('gry'),self.read('gry'))
		self.assertTrue(ergas == 0)
