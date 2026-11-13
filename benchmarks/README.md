# Q2n benchmark: pure-Python vs. pyfastcd hypercomplex products

Q2n multiplies pixels as Cayley-Dickson numbers (quaternions for 4 bands, octonions for 8, and so on). sewar ships a pure-Python implementation of that product and uses the compiled kernels of the optional [pyfastcd](https://pypi.org/project/pyfastcd/) package when it is installed (`pip install sewar[fast]`). `q2n_fastcd.py` checks that the two give the same results and measures the difference in speed.

## Running it

From a checkout of this repository:

```
pip install -r requirements.txt "pyfastcd>=1.4.0"
python benchmarks/q2n_fastcd.py            # full run, about a minute
python benchmarks/q2n_fastcd.py --quick    # smaller images, fewer rounds
python benchmarks/q2n_fastcd.py --json results.json   # also save the raw timings
```

The script prints Markdown tables like the ones below. Before timing anything it compares the products of the two backends and the Q2n value of every test image, and stops if they differ by more than `1e-12`.

## What is measured

- **Products of one block.** One pixelwise product of a 32x32 block plus one product of the block means: the hypercomplex arithmetic `_onions_quality` performs per block, and the only code that differs between the backends.
- **Complete `q2n()` calls.** The lena RGB test image (3 bands, padded to 4) and seeded random images with 8, 16 and 32 bands. Rounds alternate the backends, so background load affects both alike, and the tables give medians.
- **Ceiling.** The same `q2n()` calls with a product that costs nothing. Everything else in Q2n (per-band normalisation, means, slicing) is identical for both backends, so this is the largest speedup that any replacement of the products could reach.

## Reference results

Measured on 2026-11-12 on a shared host, single process.

- cpu: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- platform: Linux-7.2.11-arch1-1-x86_64-with-glibc2.44
- python: 3.14.8
- numpy: 2.5.4
- pyfastcd: 1.4.0 (manylinux wheel from PyPI)

Largest relative difference between the backends' products: 3.9e-16. Largest difference in Q2n: 1.1e-16.

### Hypercomplex products of one 32x32 block (microseconds)

| N | python | pyfastcd | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 141.4 | 21.9 | 6.5x |
| 8 | 646.5 | 30.6 | 21.1x |
| 16 | 2780.6 | 77.6 | 35.8x |
| 32 | 11508.0 | 216.8 | 53.1x |

### Complete q2n() calls (milliseconds)

Median [interquartile range] over interleaved rounds. "wins" counts the rounds in which pyfastcd was faster.

| image | python | pyfastcd | speedup | wins | ceiling | share of ceiling gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lena RGB 512x512 (4 after padding) | 124.1 [121.6, 127.6] | 92.6 [91.0, 96.4] | 1.34x | 25/25 | 1.52x | 74% |
| synthetic 512x512x8 | 352.0 [340.3, 358.0] | 174.3 [168.4, 177.7] | 2.02x | 15/15 | 2.25x | 91% |
| synthetic 512x512x16 | 1068.8 [1064.5, 1084.5] | 335.4 [333.8, 337.0] | 3.19x | 9/9 | 3.50x | 96% |
| synthetic 256x256x32 | 934.4 [929.3, 935.2] | 173.7 [172.5, 175.3] | 5.38x | 7/7 | 5.99x | 98% |

## Reading the results

The products themselves are 6 to 53 times faster, and the gain grows with the number of bands because the pure-Python recursion makes about N²/4 NumPy calls per block. A complete `q2n()` call gains less (1.3x for RGB, 2x for 8 bands, 5.4x for 32 bands) because the products are only part of its run time; the "ceiling" column shows how much of the remainder is shared code that this change does not touch. Grayscale images are unaffected, as with a single band the product is ordinary multiplication in both backends.
