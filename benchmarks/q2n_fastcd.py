#!/usr/bin/env python
"""Benchmark Q2n with the pure-Python and pyfastcd hypercomplex backends.

Run from a checkout of the repository, with pyfastcd installed:

	pip install "pyfastcd>=1.4.0"
	python benchmarks/q2n_fastcd.py            # full run, a few minutes
	python benchmarks/q2n_fastcd.py --quick    # smaller images, fewer rounds
	python benchmarks/q2n_fastcd.py --json results.json

The script
1. checks that both backends give the same products and the same Q2n values,
   and refuses to time anything if they do not;
2. times the hypercomplex products of one block (the only code that differs
   between the backends);
3. times complete q2n() calls.  Rounds alternate the backends so that
   background load affects both alike, and medians are reported.  A third
   variant whose products cost nothing gives the largest speedup any
   replacement of the products could reach, because the rest of Q2n is shared.
"""
from __future__ import print_function
import argparse
import json
import os
import platform
import sys
import time
import timeit

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # benchmark this checkout, not an installed sewar

import sewar  # noqa: E402
import sewar.full_ref as full_ref  # noqa: E402
from sewar import _hypercomplex as hc  # noqa: E402

RES = os.path.join(ROOT, 'sewar', 'tests', 'res')
_ZEROS = {}


def _free(o1, o2):
	"""Simulates a product that costs (almost) nothing."""
	z = _ZEROS.get(o1.shape)
	if z is None:
		z = _ZEROS[o1.shape] = np.zeros(o1.shape)
	return z


def backends():
	return {
		'python': (hc._onion_mult_python, hc._onion_mult2D_python),
		'pyfastcd': (hc._onion_mult_fastcd, hc._onion_mult2D_fastcd),
		'free': (_free, _free),
	}


def q2n_with(backend, GT, P):
	saved = full_ref._onion_mult, full_ref._onion_mult2D
	full_ref._onion_mult, full_ref._onion_mult2D = backends()[backend]
	try:
		start = time.perf_counter()
		value = sewar.q2n(GT, P)
		return time.perf_counter() - start, value
	finally:
		full_ref._onion_mult, full_ref._onion_mult2D = saved


def best_time(f, target=0.05, repeat=5):
	"""Best seconds per call, with enough calls per repeat to last about `target`."""
	start = time.perf_counter(); f(); once = time.perf_counter() - start
	number = max(1, int(target / max(once, 1e-9)))
	return min(timeit.repeat(f, number=number, repeat=repeat)) / number


def cpu_name():
	try:
		with open('/proc/cpuinfo') as f:
			for line in f:
				if line.startswith('model name'):
					return line.split(':', 1)[1].strip()
	except (IOError, OSError):
		pass
	return platform.processor() or platform.machine()


def environment():
	return {
		'python': platform.python_version(),
		'numpy': np.__version__,
		'pyfastcd': hc.pyfastcd.__version__,
		'sewar': getattr(sewar, '__version__', 'checkout'),
		'platform': platform.platform(),
		'cpu': cpu_name(),
	}


def cases(quick, rng):
	read = lambda name: np.asarray(Image.open(os.path.join(RES, name)))
	out = [('lena RGB 512x512 (4 after padding)', read('lena512color.tiff'),
		read('lena512color_noise.tiff'), 5 if quick else 25)]
	spec = ((8, 256, 3), (16, 256, 3), (32, 128, 3)) if quick else ((8, 512, 15), (16, 512, 9), (32, 256, 7))
	for N, side, rounds in spec:
		GT = rng.uniform(0, 255, (side, side, N))
		out.append(('synthetic %dx%dx%d' % (side, side, N), GT, GT + rng.normal(0, 20, GT.shape), rounds))
	return out


def check_parity(all_cases, rng):
	worst = 0.0
	for N in (2, 4, 8, 16, 32, 64):
		X, Y = rng.standard_normal((32, 32, N)), rng.standard_normal((32, 32, N))
		ref = hc._onion_mult2D_python(X, Y)
		worst = max(worst, np.abs(hc._onion_mult2D_fastcd(X, Y) - ref).max() / np.abs(ref).max())
		ref = hc._onion_mult_python(X[0, 0], Y[0, 0])
		worst = max(worst, np.abs(hc._onion_mult_fastcd(X[0, 0], Y[0, 0]) - ref).max() / np.abs(ref).max())
	worst_q = 0.0
	for _name, GT, P, _rounds in all_cases:
		worst_q = max(worst_q, abs(q2n_with('pyfastcd', GT, P)[1] - q2n_with('python', GT, P)[1]))
	if worst > 1e-12 or worst_q > 1e-12:
		sys.exit('Backends disagree (products: %.1e relative, Q2n: %.1e); not timing.' % (worst, worst_q))
	return worst, worst_q


def per_block(ws, dims, rng):
	rows = []
	for N in dims:
		X, Y = rng.standard_normal((ws, ws, N)), rng.standard_normal((ws, ws, N))
		x, y = X[0, 0].copy(), Y[0, 0].copy()
		row = {'N': N}
		for name in ('python', 'pyfastcd'):
			mult, mult2D = backends()[name]
			row[name] = best_time(lambda: (mult2D(X, Y), mult(x, y))) * 1e6
		rows.append(row)
	return rows


def end_to_end(all_cases):
	rows = []
	for name, GT, P, rounds in all_cases:
		times = {'python': [], 'pyfastcd': [], 'free': []}
		for _ in range(rounds):
			for backend in times:
				times[backend].append(q2n_with(backend, GT, P)[0] * 1e3)
		rows.append({'case': name, 'rounds': rounds, 'ms': times})
	return rows


def report(env, parity, block_rows, e2e_rows, ws):
	print('## Environment\n')
	for key in ('cpu', 'platform', 'python', 'numpy', 'pyfastcd'):
		print('- %s: %s' % (key, env[key]))
	print('\n## Parity\n')
	print('Largest relative difference between the backends\' products: %.1e. '
		'Largest difference in Q2n: %.1e.' % parity)
	print('\n## Hypercomplex products of one %dx%d block (microseconds)\n' % (ws, ws))
	print('One pixelwise product of the block plus one product of the block means, '
		'as in `_onions_quality`. Best of 5 repeats.\n')
	print('| N | python | pyfastcd | speedup |')
	print('| ---: | ---: | ---: | ---: |')
	for r in block_rows:
		print('| %d | %.1f | %.1f | %.1fx |' % (r['N'], r['python'], r['pyfastcd'], r['python'] / r['pyfastcd']))
	print('\n## Complete q2n() calls (milliseconds)\n')
	print('Median [interquartile range] over interleaved rounds. "wins" counts the rounds '
		'in which pyfastcd was faster. "ceiling" is the speedup with products that cost nothing.\n')
	print('| image | python | pyfastcd | speedup | wins | ceiling | share of ceiling gain |')
	print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
	for r in e2e_rows:
		py, fc, fr = (np.array(r['ms'][k]) for k in ('python', 'pyfastcd', 'free'))
		fmt = lambda t: '%.1f [%.1f, %.1f]' % (np.median(t), np.percentile(t, 25), np.percentile(t, 75))
		mp, mf, m0 = np.median(py), np.median(fc), np.median(fr)
		print('| %s | %s | %s | %.2fx | %d/%d | %.2fx | %.0f%% |' % (
			r['case'], fmt(py), fmt(fc), mp / mf, int((fc < py).sum()), r['rounds'],
			mp / m0, 100 * (mp - mf) / (mp - m0)))


def main():
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--quick', action='store_true', help='smaller images and fewer rounds')
	parser.add_argument('--json', metavar='PATH', help='also write raw timings to PATH')
	parser.add_argument('--ws', type=int, default=32, help='block size for the per-block timing (default 32)')
	args = parser.parse_args()
	if hc.pyfastcd is None:
		sys.exit('pyfastcd is not installed: pip install "pyfastcd>=1.4.0"')

	rng = np.random.default_rng(7)
	all_cases = cases(args.quick, rng)
	env = environment()
	parity = check_parity(all_cases, rng)
	block_rows = per_block(args.ws, (4, 8, 16, 32), rng)
	e2e_rows = end_to_end(all_cases)
	report(env, parity, block_rows, e2e_rows, args.ws)
	if args.json:
		with open(args.json, 'w') as f:
			json.dump({'environment': env, 'quick': args.quick,
				'parity': {'products_relative': parity[0], 'q2n_absolute': parity[1]},
				'per_block_us': block_rows, 'end_to_end': e2e_rows}, f, indent=1)


if __name__ == '__main__':
	main()
