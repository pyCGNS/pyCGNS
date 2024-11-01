#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.MAP as CGM
import CGNS.version

"""
  cg_dump [options] 

"""
import argparse
import os
import re
import math
import numpy
import sys

doc1 = """
  Dump tool on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  Shows the file contents.

  Using the keywords implies you have a import such as:

  import CGNS.PAT.cgnskeywords as CGK
  
""" % (
    CGNS.version.__version__
)

doc2 = """
  Examples:

  cg_dump foo.hdf
  
"""


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument(
        "-l", "--list", action="store_true", help="lists all paths of the tree"
    )
    pr.add_argument(
        "-s", "--SIDS", action="store_true", help="sorts nodes w.r.t. SIDS type order"
    )
    pr.add_argument(
        "-p", "--python", action="store_true", help="dumps a full python tree  (large)"
    )
    pr.add_argument(
        "-k",
        "--keywords",
        action="store_true",
        help="use pyCGNS keywords (requires a CGK import)",
    )
    pr.add_argument(
        "-c",
        "--pycgns",
        action="store_true",
        help="make dump a stand alone pyCGNS script",
    )
    pr.add_argument(
        "-P",
        "--paths",
        action="store_true",
        help="gives links paths as : separated string",
    )
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()
    return args


#  -------------------------------------------------------------------------
def openFile(filename, limit=True, links=["."]):
    if not CGM.probe(filename):
        return None
    if limit:
        flags = CGM.DEFAULT | CGM.S2P_NODATA
        (t, l, p) = CGM.load(filename, flags=flags, maxdata=20, lksearch=links)
    else:
        (t, l, p) = CGM.load(filename, lksearch=links)
    return (t, l, p, filename)


#  -------------------------------------------------------------------------
def main():
    args = parse()
    f_p = args.python
    f_k = args.keywords
    f_c = args.pycgns
    f_l = args.list
    f_s = args.SIDS
    l_p = args.paths
    if not l_p:
        l_p = ["."]
    else:
        if ":" in l_p:
            l_p = ":".split(l_p)
        else:
            l_p = [l_p]

    if f_c:
        f_p = True
        f_k = True

    for F in args.files:
        T = openFile(F, not f_c, links=l_p)
        if T is not None:
            print("# file:", F)
            if f_p:
                print(CGU.toString(T[0], readable=True, keywords=f_k, pycgns=f_c))
            elif f_l:
                for p in CGU.getPathsFullTree(T[0]):
                    print(p)
            else:
                CGU.prettyPrint(T[0], sort=f_s, links=T[1], paths=T[2])


if __name__ == "__main__":
    main()

# --- last line
