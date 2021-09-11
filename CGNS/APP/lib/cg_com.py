#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import os
import sys
import subprocess

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.MAP as CGM
import CGNS.version


class Context:
    def __init__(self):
        self.converter = None


def openFile(filename):
    flags = CGM.S2P_DEFAULTS | CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, maxdata=33, lksearch=['.'])
    return (t, l, p, filename)


def parseFile(filename, P, C, L):
    C.depth += 1
    R = []
    T = openFile(filename)
    LK = T[1]
    if (C.translate):
        LK = transLinks(filename, LK, P, C)
    searchLinks(LK, C, R)
    for p in R:
        L.append((filename, p))
        if (C.path):
            P.append('%s' % (p[3],))
        else:
            P.append('%s:%s' % (T[3], p[3]))
    for l in LK:
        if (l[0] == ''):
            FH = l[1]
        else:
            FH = "%s/%s" % (l[0], l[1])
        parseFile(FH, P, C, L)
    C.depth -= 1
    return P, L


def linkErrorAsString(code):
    s = ""
    if (code & CGM.S2P_LKOK):
        s += 'Link ok'
    if (code & CGM.S2P_LKFAIL):
        s += 'Link Failed:'
    if (code & CGM.S2P_LKBADSYNTAX):
        s += 'Link bad syntax'
    if (code & CGM.S2P_LKNOFILE):
        s += 'Linked-to file not found'
    if (code & CGM.S2P_LKFILENOREAD):
        s += 'Linked-to file not readable'
    if (code & CGM.S2P_LKNONODE):
        s += 'Linked-to node not found in file'
    if (code & CGM.S2P_LKLOOP):
        s += 'Link loop detected'
    if (code & CGM.S2P_LKIGNORED):
        s += 'Link ignored'
    return s


def checkString(variable, targetlist, re):
    if (variable is None):
        return False
    if (not re):
        return (variable in targetlist)
    else:
        for t in targetlist:
            if (variable.search(t) is not None):
                return True
        return False


def searchLinks(L, C, R):
    for l in L:
        add = True
        if (add or C.linklist):
            R.append(l)


def asHDFname(FA, C):
    return os.path.splitext(FA)[0] + C.exthdf


def convertInPlace(FA, FH, C):
    if not os.path.isfile(FA):
        if C.verbose:
            print('   ' * C.depth + " Error: Unreachable file: %s" % FA)
        return False
    elif not CGM.probe(FA):
        subprocess.check_output([C.converter, "-h", FA, FH])
        return True
    else:
        if C.verbose:
            print('   ' * C.depth + " Error: Mixing links to ADF and HDF files...")
        return False


def transLinks(filename, L, P, C):
    LH = []
    for l in L:
        LN = asHDFname(l[1], C)
        if (l[0] == ''):
            FA = l[1]
        else:
            FA = "%s/%s" % (l[0], l[1])
        if (C.verbose):
            print('   ' * C.depth, '->', FA)
        FH = asHDFname(FA, C)
        if (convertInPlace(FA, FH, C)):
            LH.append([l[0], LN, l[2], l[3]])
    (t, l, p) = CGM.load(filename)
    CGM.save(filename, t, links=LH)
    return LH

# --- last line
