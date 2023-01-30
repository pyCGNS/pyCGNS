#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import numpy

from ..PAT import cgnskeywords
from ..PAT import cgnsutils


def getChildren(t):
    s = set([c[0] for c in t[2]])
    cl = {}
    for c in t[2]:
        cl[c[0]] = c
    return s, cl


def compareValues(na, nb):
    va = na[1]
    vb = nb[1]
    eps = 1e-12
    if va is None and vb is not None:
        return 1
    if va is not None and vb is None:
        return 2
    if va is None and vb is None:
        return 0
    if (not isinstance(va, numpy.ndarray) or
            not isinstance(vb, numpy.ndarray)):
        return 0
    if va.dtype.char != vb.dtype.char:
        return 3
    if va.size != vb.size:
        return 4
    if va.shape != vb.shape:
        return 5
    if va.dtype.char not in ['c', 'S']:
        vd = numpy.array(numpy.abs(va - vb))
        if numpy.any(vd > eps):
            return 6
    elif va.ravel().tostring() != vb.ravel().tostring():
        return 7
    return 0


def diffAB(ta, tb, path, tag, diag, trace=False):
    diag[path] = []
    if ta[3] != tb[3]:
        diag[path].append(('CT',))
        if trace:
            print('CT %s %s' % (tag, path))
    dta = cgnsutils.getValueDataType(ta)
    dtb = cgnsutils.getValueDataType(tb)
    if dta is not cgnskeywords.MT:
        dnum = compareValues(ta, tb)
        if dnum:
            diag[path].append(('C%d' % dnum,))
            if trace:
                print('C%d %s %s' % (dnum, tag, path))
    (sa, da) = getChildren(ta)
    (sb, db) = getChildren(tb)
    a2b = sa.difference(sb)
    sn = sa.union(sb)
    for cn in sn:
        np = path + '/' + cn
        a = cn in sa
        b = cn in sb
        if not a and b:
            diag[path].append(('NA', np))
            if trace:
                print('NA %s %s' % (tag, np))
        if a and not b:
            diag[path].append(('ND', np))
            if trace:
                print('ND %s %s' % (tag, np))
        if a and b:
            diffAB(da[cn], db[cn], np, tag, diag, trace)
    if not diag[path]:
        del diag[path]
