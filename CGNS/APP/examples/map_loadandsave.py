#!/usr/bin/env python
import CGNS.MAP
import CGNS.PAT.cgnsutils as PU
import numpy
import time
import sys

numpy.set_printoptions(threshold=sys.maxint)

flags = CGNS.MAP.S2P_FOLLOWLINKS | CGNS.MAP.S2P_TRACE
start = time.clock()
(tree, links) = CGNS.MAP.load("./001Disk.hdf", flags)

node = PU.getNodeByPath(tree, "/Disk/zone1/ZoneBC/ext1/PointRange")

print("PointRange is fortran:", numpy.isfortran(node[1]), node[1], node[1].shape)

f = open("T0.py", "w+")
f.write("from numpy import *\n")
f.write("tree=")
f.write(str(tree))
f.write("\nlinks=")
f.write(str(links))
f.write("\n")
f.close()

CGNS.MAP.save("./002Disk.hdf", tree, links, flags)

end = time.clock()
print("# time =", end - start)
