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
  cg_list [options] 

"""
import argparse
import os
import re
import math
import numpy
import sys

doc1 = """
  List tool on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  The result of the list is one line per found file/node.

""" % (
    CGNS.version.__version__
)

doc2 = """
  Sort criteria:

   Each letter is a key for the sort criteria. With the key K, the pattern
   K or K+ is ascending sort on key K, K- is descending.
   Keys are N for filename, B number of bases, Z number of zones
   and so on, use option -k to have key list and -t to add list header leys.

  Examples:
  
"""


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument("-V", "--version", action="store_true", help="show version")
    pr.add_argument("-B", "--base", action="store_true", help="number of bases")
    pr.add_argument("-Z", "--zone", action="store_true", help="number of zones")
    pr.add_argument(
        "-S", "--structured", action="store_true", help="number of structured zones"
    )
    pr.add_argument(
        "-U", "--unstructured", action="store_true", help="number of unstructured zones"
    )
    pr.add_argument("-F", "--family", action="store_true", help="number of families")
    pr.add_argument("-L", "--links", action="store_true", help="number of links")
    pr.add_argument("-M", "--mega", action="store_true", help="size of file in Mb")
    pr.add_argument("-G", "--giga", action="store_true", help="size of file in Gb")
    pr.add_argument(
        "-P", "--points", action="store_true", help="total number of points"
    )
    pr.add_argument(
        "-r", "--restrict", dest="path", help="restrict all counts to path subtree"
    )
    pr.add_argument("-t", "--title", action="store_true", help="add columns header")
    pr.add_argument("-k", "--keys", action="store_true", help="show list of keys")
    pr.add_argument(
        "-s", "--sort", dest="sort", help="sort with criteria (see doc below)"
    )
    pr.add_argument("-a", "--all", action="store_true", help="alias for -hVBLZSUWK")
    pr.add_argument("-n", "--nolinks", action="store_true", help="do not follow links")
    pr.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force parse for very large files > 1Gb",
    )
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()
    return args


#  -------------------------------------------------------------------------
class Entry(object):
    _att = {
        "version": [0.0, " %1s", "V", 0, 0.0, "CGNS/HDF5 version"],
        "filename": ["", " %-1s", "N", 0, "", "Filename"],
        "nlinks": [0, " %1d", "L", 0, 0, "# of links"],
        "nbases": [0, " %1d", "B", 0, 0, "# of bases"],
        "nfamilies": [0, " %1d", "F", 0, 0, "# of families"],
        "nzones": [0, " %1d", "Z", 0, 0, "# of zones"],
        "nzones_structured": [0, " %1d", "S", 0, 0, "# of structured zones"],
        "nzones_unstructured": [0, " %1d", "U", 0, 0, "# of unstructured zones"],
        "size_mega": [0.0, " %1s", "M", 0, 0, "size of the file (Mb)"],
        "size_giga": [0.0, " %1s", "G", 0, 0, "size of the file (Gb)"],
        "npoints": [0.0, " %1d", "P", 0, 0, "number of points"],
    }
    _lvs = []

    def __init__(self):
        for h in self._att:
            self._att[h][3] = len(self._att[h][2])
            self._att[h][0] = self._att[h][4]

    def __getattr__(self, att):
        return self._att[att][0]

    def __setattr__(self, att, value):
        if att in self._att:
            dtmp = self._att
            dtmp[att][0] = value
            dtmp[att][3] = max(dtmp[att][3], len(dtmp[att][1] % value))
            super(Entry, self).__setattr__("_att", dtmp)
            if att not in self._lvs:
                self._lvs.append(att)
        else:
            super(Entry, self).__setattr__(att, value)

    @classmethod
    def keys(self):
        return zip(
            [self._att[a][2] for a in self._att], [self._att[a][5] for a in self._att]
        )

    def foundkeys(self):
        return self._lvs

    def line(self):
        return (
            [self._att[a][2] for a in self._lvs],
            [self._att[a][1] for a in self._lvs],
            [self._att[a][3] for a in self._lvs],
            tuple([self._att[a][0] for a in self._lvs]),
            self.foundkeys(),
        )


# -------------------------------------------------------------------------
def openFile(filename):
    if not CGM.probe(filename):
        return None
    flags = CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, maxdata=20)
    return (t, l, p, filename)


#  -------------------------------------------------------------------------
def parseFile(filename, A):
    if not os.path.isfile(filename):
        return None
    st = os.stat(filename)
    R = Entry()
    if not A.force and (st.st_size > 1e9):
        print("# skip file >1Gb :", filename)
        return None
    T = openFile(filename)
    if T is None:
        return None
    fmt = ""
    hdr = ""
    if A.version:
        node = CGU.getNodeByPath(
            T[0], "%s/%s" % (CGK.CGNSTree_s, CGK.CGNSLibraryVersion_s)
        )
        if node is None:
            R.version = "?.??"
        else:
            R.version = "%.2f" % (node[1].flat[0])
    if A.links:
        R.nlinks = len(T[1])
    if A.base:
        R.nbases = len(
            CGU.getAllNodesByTypeList(T[0], [CGK.CGNSTree_ts, CGK.CGNSBase_ts])
        )
    if A.family:
        R.nfamilies = len(
            CGU.getAllNodesByTypeList(
                T[0], [CGK.CGNSTree_ts, CGK.CGNSBase_ts, CGK.Family_ts]
            )
        )
    if A.zone or A.structured or A.unstructured or A.points:
        zlist = CGU.getAllNodesByTypeList(
            T[0], [CGK.CGNSTree_ts, CGK.CGNSBase_ts, CGK.Zone_ts]
        )
        if A.zone:
            R.nzones = len(zlist)
        if A.structured or A.unstructured or A.points:
            R.nzones_structured = 0
            R.nzones_unstructured = 0
            R.npoints = 0
            for z in zlist:
                zn = CGU.getNodeByPath(T[0], z + "/" + CGK.ZoneType_s)
                if CGU.stringValueMatches(zn, CGK.Structured_s):
                    if A.structured:
                        R.nzones_structured += 1
                    if A.points:
                        zz = CGU.getNodeByPath(T[0], z)
                        R.npoints += zz[1].cumprod(axis=0)[:, 0][-1]
                if CGU.stringValueMatches(zn, CGK.Unstructured_s):
                    if A.unstructured:
                        R.nzones_unstructured += 1
                    if A.points:
                        zz = CGU.getNodeByPath(T[0], z)
                        R.npoints += zz[1][0][0]
    if A.mega:
        R.size_mega = "%.2f" % (st.st_size / 1e6)
    if A.giga:
        R.size_giga = "%.2f" % (st.st_size / 1e9)
    R.filename = filename
    return R.line()


#  -------------------------------------------------------------------------
def sortResult(L, A, hdr):
    if not A.sort:
        return L
    ix = 0
    kl = [v[0] for v in Entry.keys()]
    sl = []
    while ix < len(A.sort):
        v = A.sort[ix]
        sx = None
        so = False
        if (v in kl) and (v in hdr):
            sx = hdr.index(v)
        if ((ix + 1) < len(A.sort)) and (A.sort[ix + 1] == "+"):
            ix += 1
        if ((ix + 1) < len(A.sort)) and (A.sort[ix + 1] == "-"):
            so = True
            ix += 1
        if sx is not None:
            sl.append((sx, so))
        ix += 1
    sl.reverse()
    for k, r in sl:
        L = sorted(L, key=lambda v: v[k], reverse=r)
    return L


#  -------------------------------------------------------------------------
def main():
    args = parse()

    if args.keys:
        kl = Entry.keys()
        print("# List of keys")
        for k, d in sorted(kl):
            print(k, ":", d)
        sys.exit(0)

    if args.all:
        args.title = True
        args.version = True
        args.links = True
        args.base = True
        args.zone = True
        args.unstructured = True
        args.structured = True

    files = args.files
    if args.files == []:
        import glob

        files = glob.glob("*")

    mxl = []
    hdr = None
    vls = []
    ftm = ""

    for F in files:
        try:
            R = parseFile(F, args)
            if R is not None:
                hdr = tuple(R[0])
                fmt = R[1]
                mxl.append(R[2])
                vls.append(R[3])
        except CGM.EmbeddedCHLone.CHLoneException:
            pass

    vls = sortResult(vls, args, hdr)

    if mxl != []:
        mx = numpy.array(mxl).max(axis=0)
        hdr = (
            "".join([fmt[i].split("1")[0] + str(mx[i]) + "s" for i in range(len(fmt))])
            % hdr
        )
        fmt = "".join(
            [
                fmt[i].split("1")[0] + str(mx[i]) + fmt[i].split("1")[1]
                for i in range(len(fmt))
            ]
        )

        if args.title:
            print(hdr)
        for s in vls:
            print(fmt % s)


if __name__ == "__main__":
    main()
# --- last line
