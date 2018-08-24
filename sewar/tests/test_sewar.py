from unittest import TestCase

import numpy as np
import tifffile as tif
import sewar

import os

class TestMetrics(TestCase):
	def __init__(self, *args, **kwargs):
		super(TestMetrics, self).__init__(*args, **kwargs)
		self._TEST_DIR = os.path.dirname(__file__)
		self._IMG1_PATH = os.path.join(self._TEST_DIR,'resources/lena512color.tiff')
	
	def test_mse_typical(self):
		mse = sewar.mse(tif.imread(self._IMG1_PATH),tif.imread(self._IMG1_PATH))
		self.assertTrue(mse == 0.0)

	def test_rmse_typical(self):
		rmse = sewar.rmse(tif.imread(self._IMG1_PATH),tif.imread(self._IMG1_PATH))
		self.assertTrue(rmse == 0.0)

	def test_rmse_sw_typical(self):
		rmse_sw = sewar.rmse_sw(tif.imread(self._IMG1_PATH),tif.imread(self._IMG1_PATH))
		self.assertTrue(rmse_sw == 0.0)

	def test_psnr_typical(self):
		psnr = sewar.psnr(tif.imread(self._IMG1_PATH),tif.imread(self._IMG1_PATH))
		self.assertTrue(psnr == np.inf)