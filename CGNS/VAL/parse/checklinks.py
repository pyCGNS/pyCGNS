#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
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
# Diagnostics are more detailled in this check, diags are associated with
# a link entry, the search path or with the link list, both can have more
# than one diag:
#
# -- per-link
# 100 link is ok
# 101 link is external to the source file
# 102 link is internal to the source file
#
# 200 no source node parent
# 201 no destination node
# 202 cannot have read access to destination arg file
# 203 arg file doesn't exists
# 204 arg file is not a CGNS/HDF5 file
# 205 arg file found in search path but doesn't match with arg dir
# 206 file name contains a relative path
# 207 file name contains an absolute path
# 208 file name contains dangerous characters
#
# -- per-link list
# 300 link list is ok
# 301 has links to the same node
# 302 more than one directory used
#
# 400 has duplicated entries
# 401 has conflicting entries for the same node
# 402 files have different CGNS versions
# 403 files have different name extensions
# 404 some files are found as duplicated in different paths
# 405 unreasonable link depth (more than 3)
# 406 creates an internal loop
# 407 creates an external loop
# 408 mix of relative and plain file names
#
# -- per-search path
# 500 search path is ok
# 501 has unreachable directories
# 502 has duplicated used files in different directories
#
# In the case of node masking or inconsistency, for example if the CGNSBase
# dimensions are not the same, if the ReferenceState is masked... the check
# has to be performed on the resulting merged tree. No diag related to node
# scoping is emitted by the link check
#
# -- Graph construction
# The graph representation requires a unique node identifier as an 1 to N
# index for N nodes. The convention is to start from 1 with CGNSTree_t node
# and then to count nodes during a width-first parse using an alphabetical
# order of the children names.
# Once this index is set a tuple if formed for each node with the file index
# as the first element. If a link is found, the node index is replaced by
# the target file node index. The construction algorithm is:
# * the top file is open and parsed, the node index are set.
# * an internal link is found, actual local file node index is set
# * an external link is found, the target file is added into pending files
#   list, and an entry in with the target path of the node and the node local
#   node index of its is pushed in the pending linked nodes list
# * next pending file is open and parsed the same way
# * when no more pending file exists, all pending linked nodes are processed
#   by adding the target node index as child of the parent entry
# * when no mode pending linked node exists, the graph is built
#
# -- Loop detection
# The graph is parsed with a depth-first algo, if a node index already exists
# in the current node chain of indices, there is a loop...
#
#  -------------------------------------------------------------------------
import CGNS.MAP
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnstypes as CGT
import CGNS.PAT.cgnskeywords as CGK
import math
import os


def canonicalName(filename):
    fn = os.path.expanduser(os.path.expandvars(filename))
    fn = os.path.realpath(os.path.abspath(fn))
    fn = os.path.normpath(fn)
    return fn


class CGNSGraph(object):
    pending = {}
    fileindex = []
    index = []

    def __init__(self, filename=None):
        if filename is not None:
            self.parseOneFile(filename)

    def addCanonicalFilename(self, filename):
        fn = canonicalName(filename)
        if fn not in self.fileindex:
            self.fileindex += [fn]
            return True
        return False

    def filenameIndex(self, filename):
        fn = canonicalName(filename)
        return self.fileindex.index(fn) + 1

    def parseOneFile(self, filename):
        if not self.addCanonicalFilename(filename):
            return
        flags = CGNS.MAP.S2P_NODATA | CGNS.MAP.S2P_FOLLOWLINKS
        (t, l, p) = CGNS.MAP.load(filename, flags, lksearch=["."])
        idx = self.filenameIndex(filename)
        self.index += CGU.getAllNodesAsWidthFirstIndex(t, idx)
        self.fillLinksList(idx, l)
        for el in l:
            self.parseOneFile(el[0] + "/" + el[1])
        self.solveLinks()

    def fillLinksList(self, idx, l):
        for lk in l:
            fn = canonicalName(lk[0] + os.sep + lk[1])
            print("FILE", fn)
        print(len(self.fileindex))
        for i in self.fileindex:
            print(i)

    def solveLinks(self):
        pass

    def showIndex(self, sort=False):
        if sort:
            self.index.sort()
        sz = int(math.log10(max([i[1] for i in self.index])) + 1)
        fmt = "%%.2d %%.%dd %%s" % sz
        for e in self.index:
            print(fmt % (e[0], e[1], e[2]))


for i in range(12):
    g = CGNSGraph("/tmp/CHLone-test-008-%.2d.hdf" % i)
    g.showIndex()
