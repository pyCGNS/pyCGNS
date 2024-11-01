#!/usr/bin/env python
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

doc1 = """
  Browsing and editing links on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s

  Warning: translation of ADF files is made *in place*, in other word if a link
  with the path '/A/C/D.adf' is found, its translation is '/A/C/D.hdf'. Then
  it requires correct access rights, disk place and so on...

  Warning: translation of ADF files requires the cgnsconvert tool, can be
  found on http://www.cgns.org tools
  
""" % (
    CGNS.version.__version__
)

doc2 = """
  Examples:

  * Recursively translates all ADF files (as far as you have cgnsconvert) and
  sets the linked-to file names to these in-place translated files. The
  verbose mode makes cg_link to print all parsed files:
  
  cg_link -vt 124Disk.cgns

  Link status:

  
"""

#  -------------------------------------------------------------------------
# Get a link list and a search path list and check it, return diag list
#
# A link entry is (target dir, target file, target node, local node, status)
#
# the status is the CHLone (but it is unused by this check):
# S2P_LKOK          link ok
# S2P_LKNOFILE      file not found in search path
# S2P_LKIGNORED     link ignored (unused by this check)
# S2P_LKFILENOREAD  cannot read target file
# S2P_LKNONODE      cannot find target node
#
# Diagnostics are:
#
# -- per-link
# 100  no source node parent
# 101  no destination node
# 102  cannot read destination arg file
# 103  arg file doesn't exists
# 104  arg file is not a CGNS/HDF5 file
# 105  arg file found in search path but doesn't match with arg dir
#
# -- global to the link list
# LKLOOPINTERNAL  creates an internal loop
# LKLOOPEXTERNAL  creates an external loop
# LKMIXEDPATHS    more than one directory used
# LKDUPLICATEFILE some files are found as duplicated in diffrent paths
#
# Infos are (per link):
#
# LKEXTERNALLINK external link
# LKINTERNALLINK internal link
# LKSHAREDDEST   the fileinternal link
#

import argparse


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument(
        "-t",
        "--translate",
        action="store_true",
        help="translates ADF to HDF and propagates through links",
    )
    pr.add_argument(
        "-l", "--linklist", action="store_true", help="list all link entries"
    )
    pr.add_argument(
        "-p",
        "--path",
        action="store_true",
        help="return only the path of the matching node, no filename",
    )
    pr.add_argument("-v", "--verbose", action="store_true", help="trace mode")
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()
    return args


class Context:
    def __init__(self):
        self.converter = None


def openFile(filename):
    flags = CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, maxdata=20)
    return (t, l, p, filename)


def parseFile(filename, P, C):
    C.depth += 1
    R = []
    T = openFile(filename)
    LK = T[1]
    if C.translate:
        LK = transLinks(filename, LK, P, C)
    searchLinks(LK, C, R)
    for p in R:
        if C.path:
            P.append("%s" % (p,))
        else:
            P.append("%s:%s" % (T[3], p))
    for l in LK:
        if l[0] == "":
            FH = l[1]
        else:
            FH = "%s%s%s" % (l[0], os.path.sep, l[1])
        parseFile(FH, P, C)
    C.depth -= 1
    return P


def checkString(variable, targetlist, re):
    if variable is None:
        return False
    if not re:
        return variable in targetlist
    else:
        for t in targetlist:
            if variable.search(t) is not None:
                return True
        return False


def searchLinks(L, C, R):
    for l in L:
        add = True
        if add or C.linklist:
            R.append(l[3])


def asHDFname(FA, C):
    return os.path.splitext(FA)[0] + C.exthdf


def convertInPlace(FA, FH, C):
    if not os.path.isfile(FA):
        if C.verbose:
            print("   " * C.depth + " Error: Unreachable file: %s" % FA)
        return False
    elif not CGM.probe(FA):
        subprocess.check_output([C.converter, "-h", FA, FH])
        return True
    else:
        if C.verbose:
            print("   " * C.depth + " Error: Mixing links to ADF and HDF files...")
        return False


def transLinks(filename, L, P, C):
    LH = []
    for l in L:
        LN = asHDFname(l[1], C)
        if l[0] == "":
            FA = l[1]
        else:
            FA = "%s%s%s" % (l[0], os.path.sep, l[1])
        if C.verbose:
            print("   " * C.depth, "->", FA)
        FH = asHDFname(FA, C)
        if convertInPlace(FA, FH, C):
            LH.append([l[0], LN, l[2], l[3]])
    (t, l, p) = CGM.load(filename)
    CGM.save(filename, t, links=LH)
    return LH


def main():
    args = parse()
    P = []
    C = Context()
    C.translate = args.translate
    C.exthdf = ".hdf"
    C.verbose = args.verbose
    C.depth = 0
    C.path = args.path
    C.linklist = args.linklist

    if C.translate:
        try:
            cgc = subprocess.check_output(["which", "cgnsconvert"])
        except:
            try:
                cgc = subprocess.check_output(["whence", "cgnsconvert"])
            except:
                cgc = None
        if cgc is None:
            if C.verbose:
                print("cg_link Error: Cannot find 'cgnsconvert' in the PATH")
            sys.exit(1)
        C.converter = cgc[:-1]

    if C.verbose:
        excpt = IndexError
    else:
        excpt = CGM.EmbeddedCHLone.CHLoneException
    for F in args.files:
        try:
            if C.translate:
                FA = F
                FH = asHDFname(FA, C)
                convertInPlace(FA, FH, C)
            else:
                FH = F
            if C.verbose:
                print("Start:", FH)
            parseFile(FH, P, C)
        except excpt:
            pass

    if C.linklist:
        L = set()
        for p in P:
            L.add(p)
        L = list(L)
        L.sort()
        for p in L:
            print(p)


if __name__ == "__main__":
    main()
# --- last line
