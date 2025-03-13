import numpy as np
import scipy.linalg as scla
import scipy.sparse as scsp
from scipy.sparse.linalg import splu
import os, subprocess
from fileIO import getNewFilename, readFile

class engineBase:
    def __init__(self, shape:str, shape_pars:str, contrast_squared:float,
                 n_mesh:int, quad_order:int, execfolder:str="bin",
                 datafolder:str="data", num_threads:int="auto",
                 rescale_neumann:bool=0, mass_lumped:bool=0,
                 remove_temporary_files:bool=1):
        self.shape, self.shape_pars = shape, shape_pars
        self.contrast_squared, self.n_mesh = str(contrast_squared), str(n_mesh)
        self.quad_order, self.execfolder = str(quad_order), execfolder
        self.datafolder = datafolder
        self.remove_temporary_files = remove_temporary_files
        self.rescale_neumann = str(1 * bool(rescale_neumann))
        
        self.environment = dict(os.environ)
        if num_threads == "auto":
            try:
                from multiprocessing import cpu_count
                num_threads = cpu_count()
            except:
                num_threads = 1
        self.environment["OMP_NUM_THREADS"] = str(num_threads)

        massFilename = getNewFilename(f"{datafolder}/mass", "dat")
        subprocess.run([f"{self.execfolder}/mass", self.shape, self.shape_pars,
                        self.n_mesh, self.quad_order, massFilename, str(mass_lumped)],
                       env = self.environment)
        self.energyMatrix = scsp.csr_matrix(readFile(massFilename, real = 1))
        if self.remove_temporary_files: os.remove(massFilename)
        if mass_lumped:
            self.energyMatrix = scsp.diags(self.applyEnergy(np.ones(self.halfsize)))

    @property
    def halfsize(self):
        return self.energyMatrix.shape[1]

    @property
    def size(self):
        return 2 * self.halfsize

    def getBEMMatrices(self, k):
        # must store also matrix2
        assemblerFilenames = [getNewFilename(f"{self.datafolder}/matrix1", "dat"),
                              getNewFilename(f"{self.datafolder}/matrix2", "dat"),
                              f"{self.datafolder}/foo"]
        subprocess.run([f"{self.execfolder}/assembler", self.shape,
                        self.shape_pars, self.contrast_squared, str(k), "0.",
                        self.n_mesh, self.quad_order, self.rescale_neumann]
                     + assemblerFilenames, env = self.environment)
        matrices = [readFile(f) for f in assemblerFilenames[: 2]]
        if self.remove_temporary_files:
            for f in assemblerFilenames: os.remove(f)
        return matrices[0], matrices[1]

class engineDisk(engineBase):
    def __init__(self, n_mesh:int, execfolder:str, datafolder:str="data"):
        super().__init__("circle", "1.", 20., n_mesh, 11,
                         execfolder, datafolder)

class engineCShape(engineBase):
    def __init__(self, n_mesh:int, execfolder:str, datafolder:str="data"):
        super().__init__("barbedcshape", "1.5_0.2", 20., n_mesh, 11,
                         execfolder, datafolder)

class engineKite(engineBase):
    def __init__(self, n_mesh:int, execfolder:str, datafolder:str="data"):
        super().__init__("kite", "1.3_1.5", 20., n_mesh, 11,
                         execfolder, datafolder)
