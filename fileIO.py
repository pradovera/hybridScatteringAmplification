import os
import numpy as np
from gMRI.gmri import barycentricRationalFunction, barycentricRationalFunctionMulti

def getNewFilename(base:str, end:str):
    i = 0
    filename = lambda i: "{}.{}.{}".format(base, i, end)
    while os.path.exists(filename(i)): i += 1
    with open(filename(i), 'w'): pass
    return filename(i)

def readFile(filename:str, real:bool=False, flat:bool=False):
    matrix = np.loadtxt(filename, delimiter = ",", dtype = float)
    if not real:
        matrix = matrix[:, ::2] + 1j * matrix[:, 1::2]
    if flat:
        matrix = matrix.flatten()
    return matrix

def init_solver(argv):
    allowed_domains = ["disk", "cshape", "kite"]
    if len(argv) > 1:
        domain = argv[1]
    else:
        domain = input(("Input domain (allowed values: {})"
                        "\n").format(allowed_domains))
    assert domain in allowed_domains, "domain not recognized"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    exec_path = os.path.join(dir_path, "simpleTBEM", "bin")

    if domain == "disk":
        from engine import engineDisk as engine
    elif domain == "cshape":
        from engine import engineCShape as engine
    elif domain == "kite":
        from engine import engineKite as engine
    return engine(200, exec_path)
