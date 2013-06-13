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
# Diagnostics are:
#
# -- per-link
# 100  no source node parent
# 101  no destination node
# 102  cannot read destination arg file
# 103  arg file doesn't exists
# 104  arg file is not a CGNS/HDF5 file
# 105  arg file found in search path but doesn't match with arg dir
#
# -- global to the link list
# LKLOOPINTERNAL  creates an internal loop
# LKLOOPEXTERNAL  creates an external loop
# LKMIXEDPATHS    more than one directory used
# LKDUPLICATEFILE some files are found as duplicated in diffrent paths
# 
# Infos are (per link):
#
# LKEXTERNALLINK external link
# LKINTERNALLINK internal link
# LKSHAREDDEST   the fileinternal link
#
