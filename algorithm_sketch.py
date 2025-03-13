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

def run_sketch(solver, ks, gmri_tol, q, seed = 42):
    np.random.seed(seed)
    spacesize = solver.size
    b = np.random.randn(spacesize) + 1j * np.random.randn(spacesize)

    # invert mass matrix
    mass_0 = solver.energyMatrix.todense()
    mass_0_chol = scla.cholesky(mass_0, lower = True)
    mass_chol = scla.block_diag(*([mass_0_chol] * 2))
    mass_chol_H = mass_chol.T.conj()

    # build rational surrogates
    C = lambda k: (mass_chol
                 @ np.linalg.inv(solver.getBEMMatrices(k)[0])
                 @ mass_chol_H)
    sample0 = lambda k: get_RPM_vector(C(k), b, q,
                                       renormalize = False).reshape(-1, 1)
    if q == 0:
        b_norm = np.linalg.norm(b)
        app0_base = barycentricRationalFunction(sample0, 1., np.array([np.mean(ks)]),
                                                np.ones(1), np.array([[b_norm]]))
        app0_base.sampler.samples_ortho = sample0(None) / b_norm
        app0 = barycentricRationalFunctionMulti([app0_base])
    else:
        app0 = buildgMRI(sample0, 1., ks, gmri_tol, 1e-15, bisections = "auto")
    sample1 = lambda k: get_RPM_vector(C(k), b, q, renormalize = False,
                                       post_C = True).reshape(-1, 1)
    app1 = buildgMRI(sample1, 1., ks, gmri_tol, 1e-15, bisections = "auto")

    # sweep target wavenumbers
    amplification = np.empty_like(ks)
    for i, k in enumerate(ks):
        amplification[i] = (np.linalg.norm(app1(k, None))
                          / np.linalg.norm(app0(k, None)))
        print(f"{i = }, {k = }, {amplification[i] = }")

    # store results
    filename = getNewFilename("output_sketch", "csv")
    output = np.stack([ks, amplification], axis = -1)
    np.savetxt(filename, output, delimiter = ",")

    nsamples0 = np.sum([len(a.coeffs) for a in app0.apps]) - len(app0.apps) + 1
    nsamples1 = np.sum([len(a.coeffs) for a in app1.apps]) - len(app1.apps) + 1
    print(f"rational approximation 0 required {nsamples0} samples")
    print(f"rational approximation 1 required {nsamples1} samples")
    print(f"stored output to {filename}")

if __name__ == "__main__":
    solver = init_solver(sys.argv)
    from engine import engineDisk
    assert isinstance(solver, engineDisk)
    ks = np.linspace(1., 3., 1001)
    run_sketch(solver, ks, 1e-2, 0)
    run_sketch(solver, ks, 1e-2, 1)
