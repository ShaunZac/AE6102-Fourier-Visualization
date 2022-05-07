# AE6102 - Visualizing Fourier Transforms

### Authors

Reet Mehta - 180020077

Shaun Zacharia - 180110073

### Motivation for Project

We have created a 3D visual of how functions are created from their Fourier coefficients and created a GUI that will show how a function gets closer to its actual value as we increase the number of Fourier coefficients that we are retaining in the original Fourier transform. The method of dropping higher-order Fourier coefficients is a common data compression method, and this tool can help people visualize to what extent dropping coefficients is acceptable.

Additionally, we have also implemented our own 2D FFT (Fast Fourier Transform), and sped up the code using numba.

### Deliverables
- [x] numba FFT
- [x] FFT visualizing code
- [x] interactive GUI using Traits
