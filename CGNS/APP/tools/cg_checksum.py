#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import os.path
import argparse
import re
import CGNS.PAT.cgnsutils as CGU
import CGNS.MAP as CGM
import CGNS.VAL.simplecheck as CGV
import CGNS.version

doc1 = """
  CGNS/Python checksum tool
  (part of pyCGNS distribution http://pycgns.github.com)
  pyCGNS v%s

""" % (
    CGNS.version.__version__
)

doc2 = """
  Examples:

  About checksum:
  
  The checksum is performed on actual CGNS/Python values and not
  on the CGNS/HDF5 file contents that may change due to hidden internal values.
  Checksum is performed by serial hashing of strings, one string per node
  is generated using a breadth-first algo, a node string two different
  patterns depending on it is a linked-to node or a plain node:

  plain node string = name+SIDS type+number of children+depth+value string

  linked-to node string = plain node string+link string

  value string = shape string+strides+ memory zone as binary string

  link string = destination path+file name + [actual dir]

  The value string is set to "" when None is found as value.

  The link-to string is not used when option -P (plain) is set. So that a plain
  file that is split into linked-to files would have the same checksum.

  The [actual dir] string is not used unless option -D (directory dependant)
  is set. So that the same linked-to file in two different dir would lead to
  two different checksums.
  
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
    return args


class Query(object):
    def __init__(self):
        self.userkeys = ["DEFAULT"]
        self.idlist = []


def checksumFile(filename):
    flags = CGM.S2P_DEFAULT | CGM.S2P_CHECKSUM
    (t, l, p) = CGM.load(filename, flags=flags, lksearch=["."])
    return t[1].tostring()


def main():
    args = parse()

    Q = Query()
    Q.path = args.path
    Q.flat = args.flat
    Q.verbose = args.verbose

    for F in args.files:
        try:
            print(checksumFile(F), F)
        except CGM.EmbeddedCHLone.CHLoneException:
            pass


if __name__ == "__main__":
    main()
# --- last line
