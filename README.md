# Sewar

Sewar is a python package designed to compare similarity between images using different metrics. You can check the documentation [here](http://sewar.readthedocs.io/).


## Implemented metrics
- [x] Mean Squared Error (MSE) 
- [x] Root Mean Sqaured Error (RMSE)
- [x] Peak Signal-to-Noise Ratio (PSNR) [[1]](https://ieeexplore.ieee.org/abstract/document/1284395/)
- [x] Structural Similarity Index (SSIM) [[1]](https://ieeexplore.ieee.org/abstract/document/1284395/)
- [x] Universal Quality Image Index (UQI) [[2]](https://ieeexplore.ieee.org/document/995823/)
- [x] Multi-scale Structural Similarity Index (MS-SSIM) [[3]](https://ieeexplore.ieee.org/abstract/document/1292216/)
- [x] Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) [[4]](https://hal.archives-ouvertes.fr/hal-00395027/)
- [x] Spatial Correlation Coefficient (SCC) [[5]](https://www.tandfonline.com/doi/abs/10.1080/014311698215973)
- [x] Relative Average Spectral Error (RASE) [[6]](https://ieeexplore.ieee.org/document/1304896/)
- [x] Spectral Angle Mapper (SAM) [[7]](https://ntrs.nasa.gov/search.jsp?R=19940012238)
- [ ] Visual Information Fidelity (VIF) [[8]](https://ieeexplore.ieee.org/abstract/document/1576816/)

## Installation
Just as simple as
```
pip install sewar
```
## Example usage
a simple example to use UQI
```python
>>> from sewar.metrics import uqi
>>> uqi(img1,img2)
0.9586952304831419
```
## References
[1] "Image quality assessment: from error visibility to structural similarity." 2004)<br/>
[2] "A universal image quality index." (2002)<br/>
[3] "Multiscale structural similarity for image quality assessment." (2003)<br/>
[4] "Quality of high resolution synthesised images: Is there a simple criterion?." (2000)<br/>
[5] "A wavelet transform method to merge Landsat TM and SPOT panchromatic data." (1998)<br/>
[6] "Fusion of multispectral and panchromatic images using improved IHS and PCA mergers based on wavelet decomposition." (2004)<br/>
[7] "Discrimination among semi-arid landscape endmembers using the spectral angle mapper (SAM) algorithm." (1992)<br/>
[8] "Image information and visual quality." (2006)<br/>