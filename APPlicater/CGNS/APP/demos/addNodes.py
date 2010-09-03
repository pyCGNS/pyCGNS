#!/usr/bin/env python

# [ name, value, children, type ]
import CGNS.MAP
import CGNS.PAT
import path_utils

(tree,links)=CGNS.MAP.load("./001Disk.hdf",CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE)

ref=['ReferenceState',None,[],CGNS.PAT.ReferenceState_t]
ref[2].append(['Density',dens,[],'DataArray_t'])

node=path_utils.getNodeByPath("/Disk",tree)
node[2].append(ref)

CGNS.MAP.save("./001Disk-new.hdf",CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE,tree,links)
