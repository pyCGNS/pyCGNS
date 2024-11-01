#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

try:
    import sip

    try:
        sip.setapi("QVariant", 2)
        sip.setapi("QString", 2)
    except AttributeError:
        pass
except ImportError:
    pass

import sys
import os


def main():
    prog = sys.argv[0]
    dpref = "." + os.path.sep
    if prog[0] == os.path.sep:
        dpref = os.path.dirname(prog)
    else:
        for pp in os.environ["PATH"].split(":"):
            if os.path.isfile("%s%s%s" % (pp, os.path.sep, prog)):
                dpref = os.path.dirname("%s%s%s" % (pp, os.path.sep, prog))
                break

    ppref = dpref + "/../lib/python%s/site-packages" % sys.version[:3]
    ppref = os.path.normpath(os.path.abspath(ppref))
    if ppref not in sys.path:
        sys.path.append(ppref)

    ppref = dpref + "/../Lib/site-packages"
    ppref = os.path.normpath(os.path.abspath(ppref))
    if ppref not in sys.path:
        sys.path.append(ppref)

    try:
        import numpy
    except KeyError:
        print("""CGNS.NAV: FATAL error, cannot import numpy...""")
        sys.exit(-1)

    try:
        import qtpy.QtCore
        import qtpy.QtWidgets
        import qtpy.QtGui
    except:
        print("""CGNS.NAV: FATAL error, cannot import qtpy.QtCore...""")
        sys.exit(-1)

    try:
        import vtk
    except:
        print("""CGNS.NAV: Warning, cannot import vtk...""")

    try:
        import CGNS.MAP
        import CGNS.version
    except:
        print("""CGNS.NAV: FATAL error, cannot import CGNS.MAP...""")
        sys.exit(-1)

    from CGNS.NAV.moption import Q7OptionContext as OCTXT

    doc1 = """
      Visual browsing of CGNS/HDF5 files 
      (part of pyCGNS distribution http://pycgns.github.io)
      pyCGNS v%s
    """ % (
        CGNS.version.__version__
    )

    doc2 = """
      
      The browser provides the user with many different views of the CGNS tree.
      Each view has its on on-line self-contained contextual doc (no web access)
    
    """

    import argparse
    import re

    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )

    pr.add_argument(
        "-P",
        "--profilepath",
        dest="path",
        help="override CGNSPROFILEPATH variable for profile search",
    )
    pr.add_argument(
        "-R", "--recurse", action="store_true", help="recurse open on tree view"
    )
    pr.add_argument("-l", "--last", action="store_true", help="open last file used")
    pr.add_argument(
        "-Q", "--querylist", action="store_true", help="list all known queries"
    )
    pr.add_argument(
        "-q", "--query", dest="query", help="run query and open selection view"
    )
    pr.add_argument("-g", "--graphic", action="store_true", help="open VTK view")
    pr.add_argument("-S", "--nosplash", action="store_true", help="no splash screen")
    pr.add_argument(
        "-C", "--hidecontrol", action="store_true", help="hide control window at start"
    )
    pr.add_argument("-v", "--verbose", action="store_true", help="trace mode")
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()

    class Query(object):
        def __init__(self):
            self.userkeys = ["DEFAULT"]
            self.idlist = []

    try:
        ppath = os.environ["CGNSPROFILEPATH"]
    except KeyError:
        ppath = ""

    if args.path is not None:
        ppath = args.path

    flags = (
        args.recurse,
        args.last,
        args.verbose,
        args.graphic,
        args.querylist,
        args.nosplash,
        args.hidecontrol,
    )

    files = args.files
    datasets = []

    import CGNS.NAV.script

    CGNS.NAV.script.run(sys.argv, files, datasets, flags, ppath, args.query)


if __name__ == "__main__":
    main()

# --- last line
