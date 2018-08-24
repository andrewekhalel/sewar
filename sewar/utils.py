import numpy as np


def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes"
	assert GT.dtype == P.dtype, "Supplied images have different dtypes"
