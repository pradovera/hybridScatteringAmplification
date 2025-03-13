import sys
import numpy as np
from scipy import linalg as scla
from fileIO import getNewFilename, init_solver
from gMRI.gmri import buildgMRI, barycentricRationalFunction, barycentricRationalFunctionMulti

def get_RPM_vector(C, b, N_iter, renormalize = True, post_C = False):
    N_iter = max(N_iter, 0)
    if N_iter > 0: ChC = C.T.conj() @ C
    for _ in range(N_iter):
        b = ChC @ b
        if renormalize: b /= np.linalg.norm(b)
    if post_C: return C @ b
    return b

def run_RPM(C, b, N_iter, return_vectors = False):
    b = get_RPM_vector(C, b, N_iter, not return_vectors)
    Cb = C @ b
    if return_vectors: return Cb, b
    return np.linalg.norm(Cb) / np.linalg.norm(b)

def run_sketch(solver, ks, gmri_tol, Q, seed = 42):
    np.random.seed(seed)
    spacesize = solver.size
    B = np.array([np.random.randn(spacesize) + 1j * np.random.randn(spacesize)
                                                          for _ in range(Q)]).T
    B /= np.linalg.norm(B, axis = 0)

    # invert mass matrix
    mass_0 = solver.energyMatrix.todense()
    mass_0_chol = scla.cholesky(mass_0, lower = True)
    mass_chol = scla.block_diag(*([mass_0_chol] * 2))
    mass_chol_H = mass_chol.T.conj()

    # build rational surrogate
    sample = lambda k: (mass_chol
                      @ np.linalg.solve(solver.getBEMMatrices(k)[0],
                                        mass_chol_H @ B)).reshape(-1, 1)
    app = buildgMRI(sample, 1., ks, gmri_tol, 1e-15, bisections = "auto")

    # sweep target wavenumbers
    amplification = np.empty_like(ks)
    for i, k in enumerate(ks):
        amplification[i] = np.max(np.linalg.norm(app(k).reshape(spacesize, Q),
                                                 axis = 0))
        print(f"{i = }, {k = }, {amplification[i] = }")

    # store results
    filename = getNewFilename("output_sketch_multi", "csv")
    output = np.stack([ks, amplification], axis = -1)
    np.savetxt(filename, output, delimiter = ",")

    nsamples = np.sum([len(a.coeffs) for a in app.apps]) - len(app.apps) + 1
    print(f"rational approximation required {nsamples} samples")
    print(f"kept {len(poles)} poles")
    print(f"stored output to {filename}")

if __name__ == "__main__":
    solver = init_solver(sys.argv)
    from engine import engineDisk
    assert isinstance(solver, engineDisk)
    ks = np.linspace(1., 3., 1001)
    run_sketch(solver, ks, 1e-2, 100)
