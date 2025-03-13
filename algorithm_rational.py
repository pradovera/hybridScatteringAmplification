import sys
import numpy as np
from scipy import linalg as scla
from fileIO import getNewFilename, init_solver
from gMRI.gmri import buildgMRI
from algorithm_sketch import run_RPM

def run_rational(solver, ks, gmri_tol, seed = 42):
    np.random.seed(seed)
    spacesize = solver.size
    q = 10 * int(np.ceil(1 + np.log(spacesize)))
    b = np.random.randn(spacesize) + 1j * np.random.randn(spacesize)

    # invert mass matrix
    mass_0 = solver.energyMatrix.todense()
    mass_0_chol = scla.cholesky(mass_0, lower = True)
    mass_chol = scla.block_diag(*([mass_0_chol] * 2))
    mass_chol_H = mass_chol.T.conj()

    # build rational surrogate
    sample = lambda k: (mass_chol
                      @ np.linalg.inv(solver.getBEMMatrices(k)[0])
                      @ mass_chol_H).reshape(-1, 1)
    app = buildgMRI(sample, 1., ks, gmri_tol, 1e-15, bisections = "auto")

    # sweep target wavenumbers
    amplification = np.empty_like(ks)
    for i, k in enumerate(ks):
        C = app(k).reshape(spacesize, spacesize)
        amplification[i] = run_RPM(C, np.array(b), q)
        # alternative by SVD: amplification[i] = scla.norm(C, 2)
        print(f"{i = }, {k = }, {amplification[i] = }")

    # store results
    filename = getNewFilename("output_rational", "csv")
    output = np.stack([ks, amplification], axis = -1)
    np.savetxt(filename, output, delimiter = ",")

    nsamples = np.sum([len(a.coeffs) for a in app.apps]) - len(app.apps) + 1
    print(f"rational approximation required {nsamples} samples")
    print(f"stored output to {filename}")

if __name__ == "__main__":
    solver = init_solver(sys.argv)
    from engine import engineDisk
    if isinstance(solver, engineDisk):
        ks = np.linspace(1., 3., 1001)
    else:
        ks = np.linspace(1., 5., 2001)
    run_rational(solver, ks, 1e-2)
