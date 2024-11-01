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

import os.path

doc1 = """
  CGNS/SIDS conformance checking tool on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  The result of the check is a textual diagnostic
""" % (
    CGNS.version.__version__
)

doc2 = """
  User requirements are identified by a <key>, all known
  keys can be listed with the -k option.

  Examples:

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
    pr.add_argument("-p", "--path", dest="path", help="start check at this node")
    pr.add_argument(
        "-f", "--flat", action="store_true", help="flat mode, do not recurse on tree"
    )
    pr.add_argument(
        "-l", "--diaglist", action="store_true", help="list all known diagnostics"
    )
    pr.add_argument(
        "-k",
        "--keylist",
        action="store_true",
        help="gives list of known user requirements keys (*long*)",
    )
    pr.add_argument(
        "-u", "--user", dest="user", help="check user requirements identified by <key>"
    )
    pr.add_argument(
        "-r",
        "--remove",
        dest="remove",
        help="remove the list of ids ( -r U012:U023:U001 )",
    )
    pr.add_argument(
        "-m",
        "--message",
        action="store_true",
        help="output by message id instead of path",
    )
    pr.add_argument("-v", "--verbose", action="store_true", help="trace mode")
    pr.add_argument("files", nargs=argparse.REMAINDER)

    args = pr.parse_args()
    return args


class Query(object):
    def __init__(self):
        self.userkeys = ["DEFAULT"]
        self.idlist = []


def checkRawFile(filename):
    if not os.path.isfile(filename):
        print("File not found")
        sys.exit(0)
    if not CGM.probe(filename):
        print("Cannot open file (low level HDF5 error)")
        sys.exit(0)


def openFile(filename):
    flags = CGM.S2P_DEFAULT | CGM.S2P_NODATA
    (t, l, p) = CGM.load(filename, flags=flags, lksearch=["."], maxdata=200)
    return (t, l, p, filename)


def parseFile(filename, Q):
    checkRawFile(filename)
    R = []
    T = openFile(filename)
    checkdiag = CGV.run(T[0], Q.verbose, Q.userkeys)
    CGV.showDiag(checkdiag, Q.idlist, bypath=Q.pathsort)


def main():
    args = parse()

    Q = Query()
    Q.path = args.path
    Q.flat = args.flat
    Q.diaglist = args.diaglist
    Q.keylist = args.keylist
    Q.verbose = args.verbose
    Q.pathsort = not args.message
    if args.remove is not None:
        Q.idlist = args.remove.split(":")
    if args.user is not None:
        Q.userkeys = [args.user]

    if Q.diaglist:
        ml = CGV.listdiags(Q.verbose, Q.userkeys)
        print(ml)
    elif Q.keylist:
        print("### Looking for keys, parsing PYTHONPATH may be long...")
        kl = CGV.listuserkeys(Q.verbose)
        print(kl)
    else:
        for F in args.files:
            try:
                parseFile(F, Q)
            except CGM.EmbeddedCHLone.CHLoneException:
                pass


if __name__ == "__main__":
    main()

# --- last line
