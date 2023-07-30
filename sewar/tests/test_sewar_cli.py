from .tester import Tester
import sewar

class TestMseCli(Tester):
	def test_color(self):
		args = dict(GT = self.path('clr'),
					P = self.path('clr'),
					metric= 'mse')
		mse = sewar.cli(args)
		self.assertTrue(mse == 0.0)

	def test_gray(self):
		args = dict(GT = self.path('gry'),
					P = self.path('gry'),
					metric= 'mse')
		mse = sewar.cli(args)
		self.assertTrue(mse == 0.0)

	def test_color_noise(self):
		args = dict(GT = self.path('clr'),
					P = self.path('clr_noise'),
					metric= 'mse')
		mse = sewar.cli(args)
		self.assertTrue(abs(mse - 2391.465875) < self.eps)

	def test_color_const(self):
		args = dict(GT = self.path('clr'),
					P = self.path('clr_const'),
					metric= 'mse')
		mse = sewar.cli(args)
		self.assertTrue(abs(mse - 2302.953958) < self.eps)

class TestErgasCli(Tester):
	def test_color(self):
		args = dict(GT = self.path('clr'),
					P = self.path('clr'),
					metric= 'ergas')
		ergas = sewar.cli(args)
		self.assertTrue(ergas == 0)
	
	def test_color_noise(self):
		args = dict(GT = self.path('clr'),
					P = self.path('clr_noise'),
					metric= 'ergas')
		ergas = sewar.cli(args)
		print(ergas)
		self.assertTrue(abs(ergas - 10.6068) < self.eps)

	def test_gray(self):
		args = dict(GT = self.path('gry'),
					P = self.path('gry'),
					metric= 'ergas')
		ergas = sewar.cli(args)
		self.assertTrue(ergas == 0)

class TestSsimCli(Tester):
	def test_color(self):
		args = dict(GT=self.path('clr'),
					P=self.path('clr'),
					metric='ssim',
					ws=9,
					K1=0.02,
					K2=0.05)
		ssim,_ = sewar.cli(args)
		self.assertTrue(ssim == 1)

class TestSscCli(Tester):
	def test_color(self):
		args = dict(GT=self.path('clr'),
					P=self.path('clr'),
					metric='scc',
					ws=9,
					win=[[-2,-2,-2],[-2,16,-2],[-2,-2,-2]])
		scc = sewar.cli(args)
		self.assertTrue(scc == 1)

class TestMsSsimCli(Tester):
	def test_color(self):
		args = dict(GT=self.path('clr'),
					P=self.path('clr'),
					metric='msssim',
					weights = [0.0448, 0.1333],
					ws=9,
					K1=0.05,
					K2=0.15)
		msssim = sewar.cli(args)
		self.assertTrue(msssim == 1)


class TestPSNRBCli(Tester):
	def test_gray(self):
		args = dict(GT=self.path('gry'),
					P=self.path('gry_noise'),
					metric='psnrb',
					)
		psnrb = sewar.cli(args)
		self.assertTrue(abs(psnrb - 15.0646) < self.eps)
