import sys
import numpy as np
from scipy import linalg as scla
from fileIO import getNewFilename, init_solver
from algorithm_sketch import run_RPM

def run_direct(solver, ks, seed = 42):
    np.random.seed(seed)
    spacesize = solver.size
    q = 10 * int(np.ceil(1 + np.log(spacesize)))
    b = np.random.randn(spacesize) + 1j * np.random.randn(spacesize)

    # invert mass matrix
    mass_0 = solver.energyMatrix.todense()
    mass_0_chol = scla.cholesky(mass_0, lower = True)
    mass_chol = scla.block_diag(*([mass_0_chol] * 2))
    mass_chol_H = mass_chol.T.conj()

    # sweep target wavenumbers
    amplification = np.empty_like(ks)
    for i, k in enumerate(ks):
        Ainv = np.linalg.inv(solver.getBEMMatrices(k)[0])
        C = mass_chol @ Ainv @ mass_chol_H
        amplification[i] = run_RPM(C, np.array(b), q)
        # alternative by SVD: amplification[i] = scla.norm(C, 2)
        print(f"{i = }, {k = }, {amplification[i] = }")

    # store results
    filename = getNewFilename("output_direct", "csv")
    output = np.stack([ks, amplification], axis = -1)
    np.savetxt(filename, output, delimiter = ",")

    print(f"stored output to {filename}")

if __name__ == "__main__":
    solver = init_solver(sys.argv)
    from engine import engineDisk
    if isinstance(solver, engineDisk):
        ks = np.linspace(1., 3., 1001)
    else:
        ks = np.linspace(1., 5., 2001)
    run_direct(solver, ks)
