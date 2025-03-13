import sys
import numpy as np
from scipy import linalg as scla
from scipy.interpolate import interp1d
from fileIO import getNewFilename, init_solver
from gMRI.gmri import buildgMRI
from algorithm_sketch import run_RPM

def run_hybrid(solver, ks, gmri_tol, strategy, n_b, filter_poles, seed = 42):
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
                      @ np.linalg.solve(solver.getBEMMatrices(k)[0],
                                        mass_chol_H @ b)).reshape(-1, 1)
    app = buildgMRI(sample, 1., ks, gmri_tol, 1e-15, bisections = "auto")

    # get pole locations
    poles = app.poles()
    dists = np.abs(ks.reshape(-1, 1) - poles.reshape(1, -1))

    # filter poles?
    if filter_poles:
        # keep only of poles that are "closest" to some k
        idx_keep = np.unique(np.argmin(dists, axis = 1))
        poles = poles[idx_keep]
        dists = dists[:, idx_keep]
    M_K = len(poles)

    # set up interpolation problem to get surrogate
    sampling = np.real(poles)
    if n_b == "none":
        add_pts = np.empty(0)
    elif n_b == "poles":
        add_pts = np.linspace(ks[0], ks[-1], M_K)
    elif n_b == "polestimes2":
        add_pts = np.linspace(ks[0], ks[-1], 2 * M_K)
    sampling = np.append(np.real(poles), add_pts)
    vals = np.empty(len(sampling))
    for j, k in enumerate(sampling):
        Ainv = np.linalg.inv(solver.getBEMMatrices(k)[0])
        C = mass_chol @ Ainv @ mass_chol_H
        vals[j] = run_RPM(C, np.array(b), q)
    
    if strategy == "sum":
        vander = np.abs(sampling[: M_K].reshape(-1, 1)
                      - poles.reshape(1, -1)) ** -1
        coeffs = np.linalg.solve(vander, vals[: M_K])
    elif strategy == "max":
        coeffs = vals[: M_K] * np.abs(np.imag(poles))
    if n_b in ["poles", "polestimes2"]:
        dists_add = np.abs(add_pts.reshape(-1, 1) - poles.reshape(1, -1))
        if strategy == "sum":
            ampl_base = dists_add ** -1 @ coeffs
        elif strategy == "max":
            ampl_base = np.max(coeffs / dists_add, axis = 1)
        coeffs_correct = vals[M_K :] - ampl_base

    # sweep target wavenumbers
    if strategy == "max":
        amplification = np.max(coeffs / dists, axis = 1)
    else:
        amplification = dists ** -1 @ coeffs
    if n_b in ["poles", "polestimes2"]:
        amplification += interp1d(add_pts, coeffs_correct)(ks)
    for i, k in enumerate(ks):
        print(f"{i = }, {k = }, {amplification[i] = }")

    # store results
    filename = getNewFilename("output_hybrid", "csv")
    output = np.stack([ks, amplification], axis = -1)
    np.savetxt(filename, output, delimiter = ",")

    nsamples = np.sum([len(a.coeffs) for a in app.apps]) - len(app.apps) + 1
    print(f"rational approximation required {nsamples} samples")
    print(f"kept {len(poles)} poles")
    print(f"stored output to {filename}")

if __name__ == "__main__":
    solver = init_solver(sys.argv)
    from engine import engineDisk
    if isinstance(solver, engineDisk):
        ks = np.linspace(1., 3., 1001)
        options_list = [("sum", "none", False), ("sum", "none", True),
                        ("sum", "poles", True), ("max", "poles", True),
                        ("max", "polestimes2", True)]
    else:
        ks = np.linspace(1., 5., 2001)
        options_list = [("sum", "poles", True), ("max", "poles", True),
                        ("max", "polestimes2", True)]
    for options in options_list:
        run_hybrid(solver, ks, 1e-2, *options)
