#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.MAP as CGM
import CGNS.VAL.simplecheck as CGV
import CGNS.version
from CGNS.NAV.diff import diffAB

import os.path
import sys
import argparse
import re

doc1 = """
  CGNS/HDF5 diff tool
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  The result of the diff is a textual diagnostic. Comparing trees A and B,
  in this order, the diagnostics are:

  ++ New node in A
  -- Node deleted from A
  #s CGNS/SIDS type different
  #t Value data type different
  #a Value shape different
  #v Value contents different
  
""" % (
    CGNS.version.__version__
)

doc2 = """
  Example:

$cg_diff NACA0012_1b_Block_0000_SUB.hdf NACA0012_1b.hdf 
#v /CGNSLibraryVersion
++ /NACA0012_HYB/BCFrField
++ /NACA0012_HYB/Block_0001
++ /NACA0012_HYB/S_SYM
++ /NACA0012_HYB/S_OUTFLOW
++ /NACA0012_HYB/BCSymmetryPlane
++ /NACA0012_HYB/S_PROFIL
#s /NACA0012_HYB/Block_0000/ZoneBC/bc_3/FamilyName
#t /NACA0012_HYB/Block_0000/ZoneGridConnectivity/NoMatchS
-- /NACA0012_HYB/Block_0000/ZoneGridConnectivity/NoMatchS/GlobBorder
-- /NACA0012_HYB/Block_0000/ZoneGridConnectivity/NoMatchS/GlobBorderDonor
#t /NACA0012_HYB/Block_0000/ZoneGridConnectivity/NoMatchS/GridConnectivityType

"""


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument("-p", "--path", dest="path", help="start diff at this node")
    pr.add_argument(
        "-f", "--flat", action="store_true", help="flat mode, do not recurse on tree"
    )
    pr.add_argument("-v", "--verbose", action="store_true", help="trace mode")
    pr.add_argument("files", nargs=argparse.REMAINDER)

    args = pr.parse_args()


class Query(object):
    def __init__(self):
        self.userkeys = ["DEFAULT"]
        self.idlist = []


def openFile(filename):
    flags = CGM.S2P_DEFAULT | CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, lksearch=["."], maxdata=200)
    return (t, l, p, filename)


def diagAnalysis(diag):
    paths = list(diag)
    paths.sort()
    for k in paths:
        for d in diag[k]:
            ldiag = ""
            if d[0] == "NA":
                ldiag = "++"
            if d[0] == "ND":
                ldiag = "--"
            if d[0] in ["CT"]:
                ldiag = "#t"
            if d[0] in ["C3", "C1", "C2"]:
                ldiag = "#a"
            if d[0] in ["C6", "C7"]:
                ldiag = "#v"
            if d[0] in ["C4", "C5"]:
                ldiag = "#s"
            if len(d) > 1:
                print(ldiag, d[1])
            else:
                print(ldiag, k)
    return ldiag


def main():
    args = parse()
    Q = Query()
    Q.path = args.path
    Q.flat = args.flat
    Q.verbose = args.verbose

    if len(args.files) != 2:
        print("cg_diff requires exactly two files to check")
        sys.exit(0)

    try:
        R1 = openFile(args.files[0])
    except CGM.EmbeddedCHLone.CHLoneException:
        print("cg_diff cannot open file [%s]" % args.files[0])
        sys.exit(0)

    try:
        R2 = openFile(args.files[1])
    except CGM.EmbeddedCHLone.CHLoneException:
        print("cg_diff cannot open file [%s]" % args.files[1])
        sys.exit(0)

    diag = {}
    diffAB(R1[0], R2[0], "", "A", diag, False)

    diagAnalysis(diag)


if __name__ == "__main__":
    main()
# --- last line
