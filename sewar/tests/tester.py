from unittest import TestCase
import os
import numpy as np
from PIL import Image

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
		return np.asarray(Image.open(os.path.join(self.RESOURCES_DIR,self.IMAGES[key])))

	def path(self,key):
		return os.path.join(self.RESOURCES_DIR,self.IMAGES[key])