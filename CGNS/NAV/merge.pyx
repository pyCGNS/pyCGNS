#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from ..PAT import cgnsutils as CGU

def mergeAB(ta, tb, tc, tag, diag, pfxA, pfxB, trace=False):
    paths = CGU.getAllPaths(ta)
    tc[2] = []
    for cta in ta[2]:
        tc[2].append(CGU.nodeCopy(cta))
    for path in paths:
        if path in diag:
            pdg = diag[path]
            for dg in pdg:
                if dg[0] == 'NA':
                    nt = CGU.getNodeByPath(tb, dg[1])
                    nd = CGU.nodeCopy(nt, str(nt[0] + pfxB))
                    np = CGU.getNodeByPath(tc, path)
                    CGU.setChild(np, nd)
                elif dg[0] == 'ND':
                    nt = CGU.getNodeByPath(ta, dg[1])
                    nd = CGU.nodeCopy(nt, str(nt[0] + pfxA))
                    np = CGU.getNodeByPath(tc, path)
                    CGU.setChild(np, nd)
                else:
                    nt = CGU.getNodeByPath(ta, path)
                    nt[0] = str(nt[0] + pfxA)
    return tc
