#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils    as CGU


def mergeTrees(treelist):
    """Merge a list of CGNS/Python trees into a single one
    Actual trees are modified
    List order is meaningful: the first tree is set as the master tree,
    if a conflict occurs and the FORCEFIRST flag is on, the master tree
    overwrites the slave tree...
    There is no loop detection"""
    if (len(treelist) < 2): return treelist[0]
    tmaster = treelist[0]
    for tslave in treelist[1:]:
        tmaster = mergeTwoTrees(tmaster, tslave)
    return tmaster


def mergeTwoTrees(TM, TS, path=''):
    TR = [TM[0], TM[1], [], TM[3]]
    C_M = set(CGU.childrenNames(TM))
    C_S = set(CGU.childrenNames(TS))
    C_L = C_M.symmetric_difference(C_S)
    for C in C_L:
        # print 'add   ',path+'/'+C
        if (C in C_M):
            T = TM
        else:
            T = TS
        C_N = CGU.hasChildNode(T, C)
        TR[2].append(C_N)
    C_L = C_M.intersection(C_S)
    for C in C_L:
        # print 'merge ',path+'/'+C
        TR[2].append(mergeTwoTrees(CGU.hasChildNode(TM, C),
                                   CGU.hasChildNode(TS, C),
                                   path + '/' + C))
    return TR

# --- last line
