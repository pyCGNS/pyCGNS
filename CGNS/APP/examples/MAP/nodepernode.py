#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.MAP
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

import time

# - load a single node in a tree
#   Nothing else but the target node and its ancestors are load
#   You load what you need, see below to get tree structure without data

path = "/Disk/zone1/GridCoordinates/CoordinateZ"
(tree, lk) = CGNS.MAP.load("data/T0.cgns", CGNS.MAP.S2P_DEFAULT, 0, path, [], None)
print(tree)
print("-" * 50)

updict = {}
path = "/Disk/zone1/GridCoordinates/CoordinateZ"
updict[path] = CGU.getValueByPath(tree, path)
path = "/Disk/zone2/GridCoordinates/CoordinateX"
updict[path] = CGU.getValueByPath(tree, path)

(tree, lk) = CGNS.MAP.load("data/T0.cgns", CGNS.MAP.S2P_DEFAULT, 0, None, [], updict)
print(tree)
print("-" * 50)

# - load only up to argument depth
#   Useful in case you only want the layout of the tree, if you want to
#   make links to already existing trees
#   If you set depth to 0 there is no depth limit
#   A level of 1 means only CGNSTree_t
#   A level of 2 means CGNSLibraryVersion_t, CGNSBase_t...
#   and so on
depth = 3
(tree, lk) = CGNS.MAP.load("data/T0.cgns", CGNS.MAP.S2P_DEFAULT, depth, None, [], None)
print(tree)
print("-" * 50)

# - do not load data but only tree skeletton
#   All data would be MT (no data, no data type)
flags = CGNS.MAP.S2P_DEFAULT | CGNS.MAP.S2P_NODATA
(tree, lk) = CGNS.MAP.load("data/T0.cgns", flags, 0, None, [], None)
print
tree

# find all zones
zlist = CGU.getAllNodesByTypeList2([CGK.Zone_ts], tree)

# load zone data
for z in zlist:
    print("-" * 50, z)
    (tree, lk) = CGNS.MAP.load("data/T0.cgns", CGNS.MAP.S2P_DEFAULT, 0, z, [], None)
    print(tree)

# --- last line
