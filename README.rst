Sewar
=====

Sewar is a python package designed to compare similarity between images
using different metrics. You can check the documentation `here`_.

Implemented metrics
-------------------

-  [x] Mean Squared Error (MSE)
-  [x] Root Mean Sqaured Error (RMSE)
-  [x] Peak Signal-to-Noise Ratio (PSNR) `[1]`_
-  [x] Structural Similarity Index (SSIM) `[1]`_
-  [x] Universal Quality Image Index (UQI) `[2]`_
-  [x] Multi-scale Structural Similarity Index (MS-SSIM) `[3]`_
-  [x] Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
   `[4]`_
-  [x] Spatial Correlation Coefficient (SCC) `[5]`_
-  [x] Relative Average Spectral Error (RASE) `[6]`_
-  [x] Spectral Angle Mapper (SAM) `[7]`_
-  [ ] Visual Information Fidelity (VIF) `[8]`_

Installation
------------

Just as simple as

::

   # install dependencies
   pip install numpy scipy
   # install sewar
   pip install --index-url https://test.pypi.org/simple/ sewar

Example usage
-------------

a simple example to use UQI

.. code:: python

   >>> from sewar.metrics import uqi
   >>> uqi(img1,img2)
   0.9586952304831419

References
----------

[1] “Image quality assessment: from error visibility to structural
similarity.” 2004) [2] “A universal image quality index.” (2002) [3]
“Multiscale structural similarity for image quality assessment.” (2003)
[4] “Quality of high resolution synthesised images: Is there a simple
criterion?.” (2000) [5] “A wavelet transform method to merge Landsat TM
and SPOT panchromatic data.” (1998) [6] “Fusion of multispectral and
panchromatic images using improved IHS and PCA mergers based on wavelet
decomposition.” (2004) [7] “Discrimination among semi-arid landscape
endmembers using the spectral angle mapper (SAM) algorithm.” (1992) [8]
“Image information and visual quality.” (2006)

.. _here: http://sewar.readthedocs.io/
.. _[1]: https://ieeexplore.ieee.org/abstract/document/1284395/
.. _[2]: https://ieeexplore.ieee.org/document/995823/
.. _[3]: https://ieeexplore.ieee.org/abstract/document/1292216/
.. _[4]: https://hal.archives-ouvertes.fr/hal-00395027/
.. _[5]: https://www.tandfonline.com/doi/abs/10.1080/014311698215973
.. _[6]: https://ieeexplore.ieee.org/document/1304896/
.. _[7]: https://ntrs.nasa.gov/search.jsp?R=19940012238
.. _[8]: https://ieeexplore.ieee.org/abstract/document/1576816/