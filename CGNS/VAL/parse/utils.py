#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import numpy as NPY


def transformCheckValues(data, cdim):
    if int(cdim) not in (1, 2, 3):
        return False
    adata = map(abs, data)
    if (cdim == 1) and (1 not in adata):
        return False
    if (cdim == 2) and (set(adata) != set((1, 2))):
        return False
    if (cdim == 3) and (set(adata) != set((1, 2, 3))):
        return False
    return True


def transformAsVector(data, cdim):
    v = [None] * cdim
    for n in range(cdim):
        v[n] = [0, 0, 0]
        v[n][abs(data[n]) - 1] = 1 * NPY.sign(data[n])
    return v


def transformIsRightHanded(data, cdim):
    return transformIsDirect(data, cdim)


def transformIsDirect(data, cdim):
    if not transformCheckValues(data, cdim):
        return False
    v = transformAsVector(data, cdim)
    if cdim == 3:
        return NPY.all(v[2] == NPY.cross(v[0], v[1]))
    return True
