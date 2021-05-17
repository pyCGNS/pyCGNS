#!/usr/bin/env python

# [ name, value, children, type ]
import CGNS.MAP
import CGNS.PAT
import CGNS.PAT.cgnsutils

(tree, links) = CGNS.MAP.load(
    "./001Disk.hdf", CGNS.MAP.S2P_FOLLOWLINKS | CGNS.MAP.S2P_TRACE
)

ref = ["ReferenceState", None, [], CGNS.PAT.ReferenceState_t]
ref[2].append(["Density", dens, [], "DataArray_t"])

node = CGNS.PAT.cgnsutils.getNodeByPath(tree, "/Disk")
node[2].append(ref)

CGNS.MAP.save(
    "./001Disk-new.hdf", CGNS.MAP.S2P_FOLLOWLINKS | CGNS.MAP.S2P_TRACE, tree, links
)
