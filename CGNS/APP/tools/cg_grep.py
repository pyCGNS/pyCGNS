#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.MAP as CGM
import CGNS.version

doc1 = """
  Grep tool on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  The result of the grep is one line per found file/node, with the
  syntax <file>:<node-path>
""" % (
    CGNS.version.__version__
)

doc2 = """
  Examples:

  cg_grep -n 'HybridBaseMerged' *
  - find all nodes with the name 'HybridBaseMerged'

  cg_grep -sn 'HybridBaseMerged' *
  - find he first file with a node having the name 'HybridBaseMerged'

  cg_grep -ct 'GridConnectivityProperty_t' *
  - find all GridConnectivityProperty_t nodes but no not continue to parse
    their children nodes
    
  cg_grep -en '[Aa]ngle' naca0012.hdf
  - find all nodes with 'Angle' or 'angle' as substring in of the node name

  cg_grep -en '^[Aa]ngle$' naca0012.hdf
  - find all nodes with 'Angle' or 'angle' as the node name

  cg_grep -sel '.*' *
  - find all links in all CGNS/HDF5 files in current dir, stop at first
    file found. The first file containing at least one link is the result.
  
"""

import argparse
import re


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument("-n", "--name", dest="name", help="grep on node name")
    pr.add_argument("-t", "--type", dest="sidstype", help="grep on SIDS type name")
    pr.add_argument(
        "-l", "--linkpath", dest="linkpath", help="grep on a path of a link"
    )
    pr.add_argument(
        "-f", "--linkfile", dest="linkfile", help="grep on a file of a link"
    )
    pr.add_argument(
        "-c",
        "--cut",
        action="store_true",
        help="leaf only, do not propagate to subtrees (cut on find)",
    )
    pr.add_argument(
        "-e",
        "--regexp",
        action="store_true",
        help="args can contain regular expressions",
    )
    pr.add_argument("-s", "--stop", action="store_true", help="stop at first found")
    pr.add_argument(
        "-p",
        "--path",
        action="store_true",
        help="return only the path of the matching node, no filename",
    )
    pr.add_argument("files", nargs=argparse.REMAINDER)

    args = pr.parse_args()
    return args


class Query(object):
    def __init__(self):
        self.name = None
        self.sidstype = None
        self.linkpath = None
        self.linkfile = None
        self.regexp = False

    def initialize(self):
        if self.regexp:
            if self.name is not None:
                self.name = re.compile(self.name)
            if self.sidstype is not None:
                self.sidstype = re.compile(self.sidstype)
            if self.linkpath is not None:
                self.linkpath = re.compile(self.linkpath)
            if self.linkfile is not None:
                self.linkfile = re.compile(self.linkfile)


def openFile(filename):
    flags = CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, maxdata=20)
    return (t, l, p, filename)


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


def searchNodes(T, Q, R):
    paths = CGU.getAllPaths(T)
    for path in paths:
        skip = False
        if Q.cut:
            for cpath in R:
                if cpath in path:
                    skip = True
        if not skip:
            add = checkString(Q.name, CGU.getPathToList(path, nofirst=True), Q.regexp)
            add |= checkString(Q.sidstype, CGU.getPathAsTypes(T, path), Q.regexp)
            if add:
                R.append(path)
                if Q.stop:
                    return


def searchLinks(L, Q, R):
    for l in L:
        add = checkString(Q.linkpath, [l[2]], Q.regexp)
        add |= checkString(Q.linkfile, [l[1]], Q.regexp)
        if add:
            R.append(l[3])
            if Q.stop:
                return


def parseFile(filename, P, Q):
    R = []
    T = openFile(filename)
    searchNodes(T[0], Q, R)
    searchLinks(T[1], Q, R)
    for p in R:
        if Q.path:
            P.append("%s" % (p,))
        else:
            P.append("%s:%s" % (T[3], p))
        if Q.stop:
            break
    return P


def main():
    args = parse()

    Q = Query()
    Q.name = args.name
    Q.sidstype = args.sidstype
    Q.linkpath = args.linkpath
    Q.linkfile = args.linkfile
    Q.cut = args.cut
    Q.path = args.path
    Q.stop = args.stop
    Q.regexp = args.regexp
    Q.initialize()

    P = []

    for F in args.files:
        try:
            parseFile(F, P, Q)
            if P and Q.stop:
                break
        except CGM.EmbeddedCHLone.CHLoneException:
            pass

    for p in P:
        print(p)


if __name__ == "__main__":
    main()
# --- last line
