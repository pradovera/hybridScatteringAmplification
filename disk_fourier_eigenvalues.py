import numpy as np
from scipy.linalg import block_diag
from scipy.special import jv, hankel1
from match_1pEVP.match_1pevp.nonparametric import beyn
from fileIO import getNewFilename

def run_fourier_eigenvalues(contrast:float):
    min_index = 0
    max_index = 20
    center = 2
    radius = 1.25

    def jvp(i, k, idx = 0):
        if idx == 0: return jv(i, k)
        if idx == 1: return .5 * (jvp(i - 1, k) - jvp(i + 1, k))
        raise Exception("not implemented")
    def h1vp(i, k, idx = 0):
        if idx == 0: return hankel1(i, k)
        if idx == 1: return .5 * (h1vp(i - 1, k) - h1vp(i + 1, k))
        raise Exception("not implemented")

    # define NLEVP matrix
    matrix_single = lambda k, i: np.array([[h1vp(i, k), jvp(i, contrast * k)],
                                           [k * h1vp(i, k, 1), contrast * k * jvp(i, contrast * k, 1)]])
    def matrix(k):
        return block_diag(*[matrix_single(k, i) for i in range(min_index, max_index + 1)])
    
    # compute eigenvalues
    space_size = 2 * (max_index - min_index + 1)
    lhs = rhs = np.eye(space_size)
    N_quad = 5000
    eigs = beyn(matrix, center, radius, lhs, rhs, N_quad, 1e-8, 5)
    
    # store results
    filename = getNewFilename("output_fourier", "csv")
    eigsRI = np.stack([np.real(eigs), np.imag(eigs)], axis = -1)
    np.savetxt(filename, eigsRI, delimiter = ",")

    print(f"stored output to {filename}")

if __name__ == "__main__":
    run_fourier_eigenvalues(20. ** .5)

