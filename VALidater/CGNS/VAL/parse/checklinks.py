#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK

#  -------------------------------------------------------------------------
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
# -- Loop detection 
