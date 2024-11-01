#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import getopt
import sys

from CGNS import version as __vid__
import CGNS.VAL.simplecheck as CGV
import CGNS.MAP
from CGNS.VAL import __vid__


def usage():
    print(
        """CGNS.VAL v%s
  
  CGNS.VAL [options] file.hdf
  
  -p <path>   : Start check at this node
  -f          : Flat mode, do not recurse on tree
  -u <key>    : Check user requirements identified by <key>
  -k          : Gives list of known user requirements keys (*long*)
  -l          : List all known diagnostics
  -r <idlist> : remove the list of ids ( -r U012:U023:U001 )
  -m          : Output by message id instead of path
  -h          : help
  -v          : verbose (trace)

  User requirements are identified by a <key>, all known
  keys can be listed with the -k option.
  See documentation for more details.
  """
        % __vid__
    )
    sys.exit(-1)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "klmfhvu:r:p:")
    except getopt.GetoptError:
        usage()

    verbose = False
    mlist = False
    klist = False
    recurse = True
    userlist = []
    pathsort = True
    olist = []
    idlist = []

    for o, v in opts:
        if o == "-k":
            klist = True
        if o == "-l":
            mlist = True
        if o == "-f":
            recurse = False
        if o == "-m":
            pathsort = False
        if o == "-p":
            path = v
        if o == "-v":
            verbose = True
        if o == "-u":
            userlist += [v]
        if o == "-r":
            idlist = v.split(":")
        if o in ("-h", "--help"):
            usage()

    try:
        filename = args[0]
    except IndexError:
        if (not mlist) and (not klist):
            usage()

    if not userlist:
        userlist = ["DEFAULT"]

    if verbose:
        print("### CGNS.VAL v%s" % __vid__)

    if mlist:
        ml = CGV.listdiags(verbose, userlist)
        print(ml)
    elif klist:
        print("### Looking for keys, parsing PYTHONPATH may be long...")
        kl = CGV.listuserkeys(verbose)
        print(kl)
    else:
        if verbose:
            print("### Loading file [%s]" % filename)
        (tree, links, paths) = CGNS.MAP.load(
            filename,
            flags=CGNS.MAP.S2P_DEFAULTS | CGNS.MAP.S2P_NODATA,
            lksearch=["."],
            maxdata=200,
        )
        checkdiag = CGV.run(tree, verbose, userlist)
        CGV.showDiag(checkdiag, idlist, bypath=pathsort)


if __name__ == "__main__":
    main()

# --- last line
