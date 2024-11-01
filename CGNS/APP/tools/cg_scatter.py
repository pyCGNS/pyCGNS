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
from CGNS.APP.lib.mergeTrees import mergeTrees

"""
  cg_scatter [options] 

"""
import argparse
import os

doc1 = """
  Scatter tool on CGNS/HDF5 files
  (part of pyCGNS distribution http://pycgns.github.io)
  pyCGNS v%s
  
  Splits the file into several files linked together
  
""" % (
    CGNS.version.__version__
)

doc2 = """
  Examples:

  cg_scatter -z verybigfile.hdf
  
"""


def parse():
    pr = argparse.ArgumentParser(
        description=doc1,
        epilog=doc2,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file1 file2 ...",
    )
    pr.add_argument(
        "-z", "--zone", action="store_true", help="use the zone a sub-tree selector"
    )
    pr.add_argument("files", nargs=argparse.REMAINDER)
    args = pr.parse_args()
    return args


#  -------------------------------------------------------------------------
def distributeZones(filename):
    print("# cg_scatter: per zone distribution for %s" % filename)
    tt = []
    flags = CGM.NODATA | CGM.DEFAULT
    print("#           :\n#           : load zone list")
    (t, l, p) = CGM.load(filename, flags=flags, maxdata=20, depth=3)
    print("#           : distribution completion")
    pl = [CGK.CGNSTree_ts, CGK.CGNSBase_ts, CGK.Zone_ts]
    sl = set(CGU.getPathsByTypeOrNameList(t, pl))
    kl = set(CGU.getPathsFullTree(t))
    rl = list(kl.difference(sl))
    vl = []
    for rp in list(rl):
        vl.append(CGU.getPathListCommonAncestor(list(sl) + [rp]))
    vl = list(set(vl))
    vl.remove("/")
    for v in set(vl):
        rl.remove(v)
    print("#           : found %d zones" % len(sl))
    filebase = os.path.splitext(filename)[0]
    filemaster = "%s-TOP.hdf" % (filebase,)
    lk = []
    for z in list(sl):
        zn = CGU.getPathLeaf(z)
        filezone = ("%s-%s.hdf" % (filebase, zn)).replace(" ", "_")
        print("#           :\n#           : load zone %s" % zn)
        lk.append(["", "", z, "", filezone, z])
        (t, l, p) = CGM.load(filename, path=z)
        print("#           : save subfile [%s]" % filezone)
        CGM.save(filezone, t)
    print("#           : load TOP without zones")
    for pp in rl:
        (t, l, p) = CGM.load(filename, path=pp)
        tt.append(t)
    master_tree = mergeTrees(tt)
    for zn in list(sl):
        bn = CGU.getNodeByPath(master_tree, CGU.getPathAncestor(zn))
        CGU.setAsChild(bn, [CGU.getPathLeaf(zn), None, [], CGK.Zone_ts])
    print("#           :\n#           : save TOP file [%s]" % filemaster)
    CGM.save(filemaster, master_tree, links=lk, flags=CGM.DEFAULTS | CGM.TRACE)


#  -------------------------------------------------------------------------
def main():
    args = parse()
    f_z = args.zone

    for F in args.files:
        if f_z:
            distributeZones(F)


if __name__ == "__main__":
    main()

# --- last line
