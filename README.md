# hybridScatteringAmplification
Reproducibility code for paper on hybrid rational surrogate modeling for field amplification of scattering problems.

This repository contains minimal code for running the numerical examples in the paper:

D. Pradovera, R. Hiptmair, and I. Perugia, _Surrogate modeling of resonant behavior in scattering problems through adaptive rational approximation and sketching_, (2025)

Preprint link: [https://arxiv.org/abs/2503.10194](https://arxiv.org/abs/2503.10194)


## Python requirements:
* sys
* os
* subprocess
* numpy
* scipy

*Tested on Python 3.10.12 with numpy v1.26.3 and scipy v1.12.0*


## Setup
First, you'll need to install the submodules. You may do this through
```sh
git submodule update --init --recusive
```

Then you'll need to compile `simpleTBEM`. For this, move to the `simpleTBEM` folder and follow the instruction in the `README` file there.

*Tested on Ubuntu 22.04 with c++ v11.4.0 and default `simpleTBEM` compiler options*

The executables `assembler` and `mass` should be automatically created within the `simpleTBEM/bin` folder.

**If you wish to place the executables elsewhere, you'll have to edit the `exec_path` variable within `fileIO.init_solver`!**

If you would like to install any of the submodules `gMRI` or `match_1pEVP`, follow the instruction in the `README` files within their respective folders.

Installing `gMRI` allows you to replace the local import
```python
from gMRI.gmri import ...
```
with the global import
```python
from gmri import ...
```
in files `algorithm_hybrid.py`, `algorithm_rational.py`, `algorithm_sketch.py`, `algorithm_sketch_multi.py`, and `fileIO.py`.

Installing `match_1pEVP` allows you to replace the local import
```python
from match_1pEVP.match_1pevp.nonparametric import beyn
```
with the global import
```python
from match_1pevp.nonparametric import beyn
```
in file `disk_fourier_eigenvalues.py`.


## Running

Then you can run any of the examples from the paper:
```sh
python algorithm_${kind}.py ${scatterer_type}
```

The placeholder `${kind}` should be one of
* `direct`, for Algorithm 1;
* `rational`, for Algorithm 2;
* `sketch`, for Algorithm 3;
* `sketch_multi`, for Algorithm 3, simplified version with $q=0$ and $Q$ different sketching vectors;
* `hybrid`, for Algorithm 4.

The placeholder `${scatterer_type}` should be one of
* `disk`, for a disk scatterer, reproducing the results in Section 6.1;
* `cshape`, for a C-shaped scatterer, reproducing the results in Section 6.2;
* `kite`, for a kite-shaped scatterer, reproducing the results in Section 6.2.

If left empty, the user is prompted to input `${scatterer_type}` at runtime.

You can also compute the analytical complex eigenvalues of the unit disk as in the paper:
```sh
python disk_fourier_eigenvalues.py
```
