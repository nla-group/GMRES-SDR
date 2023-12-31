# GMRES with sketching and deflated restarting

This repository contains files accompanying the [paper](https://arxiv.org/abs/2311.14206) listed at the bottom of this page. More specifically, the repository contains implementations of GMRES-SDR, GMRES-DR, and GCRO-DR. 

Run `test_stokes_single.m` and `test_stokes_multiple.m` to reproduce the two plots in Figure 6.1 of the paper.

GMRES_SDR requires MATLAB's Signal Processing Toolbox for the `dct()` function when the default sketching operator (subsampled discrete cosine transform) is used. It is possible to use other sketching operators by passing a `param.hS` function handle to `gmres_sdr.m`.

## Reference 
```
@techreport{BGS23,
  title   = {GMRES with randomized sketching and deflated restarting},
  author  = {Burke, Liam and G\"{u}ttel, Stefan and Soodhalter, Kirk},
  year    = {2023},
  number  = {arXiv:2311.14206},
  pages   = {22},
  institution = {arXiv}, 
  type    = {arXiv EPrint},
  url     = {https://arxiv.org/abs/2311.14206},
}
```
