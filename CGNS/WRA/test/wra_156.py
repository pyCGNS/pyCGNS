#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sy
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
from __future__ import print_function

import CGNS.WRA.mll as Mll
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print('CGNS.WRA.mll', '#156 - connectivities +CPEX 0039')


# ----------------------------------------------------------------------
def acube(im=3, jm=5, km=7, offset=0):
    # inverse k/i in order to get correct order in ADF file
    x = NPY.zeros((km, jm, im), 'd')
    y = NPY.zeros((km, jm, im), 'd')
    z = NPY.zeros((km, jm, im), 'd')
    for i in range(im):
        for j in range(jm):
            for k in range(km):
                x[k, j, i] = i + (im - 1) * offset
                y[k, j, i] = j
                z[k, j, i] = k
    return (x, y, z)


c01 = acube()
c02 = acube(offset=1)
c03 = acube(offset=2)

# ------------------------------------------------------------------------

a = Mll.pyCGNS('tmp/testmll156.hdf', Mll.MODE_WRITE)
a.base_write('Base#1', 3, 3)
a.zone_write(1, 'Zone 01', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(1, 'Zone 02', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(1, 'Zone 03', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.coord_write(1, 1, CK.RealDouble, CK.CoordinateX_s, c01[0])
a.coord_write(1, 1, CK.RealDouble, CK.CoordinateY_s, c01[1])
a.coord_write(1, 1, CK.RealDouble, CK.CoordinateZ_s, c01[2])
a.coord_write(1, 2, CK.RealDouble, CK.CoordinateX_s, c02[0])
a.coord_write(1, 2, CK.RealDouble, CK.CoordinateY_s, c02[1])
a.coord_write(1, 2, CK.RealDouble, CK.CoordinateZ_s, c02[2])
a.coord_write(1, 3, CK.RealDouble, CK.CoordinateX_s, c03[0])
a.coord_write(1, 3, CK.RealDouble, CK.CoordinateY_s, c03[1])
a.coord_write(1, 3, CK.RealDouble, CK.CoordinateZ_s, c03[2])
a.boco_write(1, 1, 'BC 01', 12, 4, 2, NPY.array([[1, 1, 1], [3, 5, 1]]))
a.boco_write(1, 1, 'BC 02', 12, 4, 2, NPY.array([[1, 1, 7], [3, 5, 7]]))
t = a.conn_1to1_write(1, 1, 'Z-Conn#01>02:03', 'Base#2/Zone 03',
                      NPY.array([[3, 1, 1], [3, 5, 7]]),
                      NPY.array([[1, 1, 1], [1, 5, 7]]),
                      NPY.array([1, 2, 3]))
a.error_print()

a.base_write('Base#2', 3, 3)
a.zone_write(2, 'Zone 01', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(2, 'Zone 02', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(2, 'Zone 03', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.coord_write(2, 1, CK.RealDouble, CK.CoordinateX_s, c01[0])
a.coord_write(2, 1, CK.RealDouble, CK.CoordinateY_s, c01[1])
a.coord_write(2, 1, CK.RealDouble, CK.CoordinateZ_s, c01[2])
a.coord_write(2, 2, CK.RealDouble, CK.CoordinateX_s, c02[0])
a.coord_write(2, 2, CK.RealDouble, CK.CoordinateY_s, c02[1])
a.coord_write(2, 2, CK.RealDouble, CK.CoordinateZ_s, c02[2])
a.coord_write(2, 3, CK.RealDouble, CK.CoordinateX_s, c03[0])
a.coord_write(2, 3, CK.RealDouble, CK.CoordinateY_s, c03[1])
a.coord_write(2, 3, CK.RealDouble, CK.CoordinateZ_s, c03[2])
a.boco_write(2, 1, 'BC 01', 12, 4, 2, NPY.array([[1, 1, 1], [3, 5, 1]]))
a.boco_write(2, 1, 'BC 02', 12, 4, 2, NPY.array([[1, 1, 7], [3, 5, 7]]))
t = a.conn_1to1_write(2, 1, 'Z-Conn#02>09:51', 'Base#9/Zone 51',
                      NPY.array([[3, 1, 1], [3, 5, 7]]),
                      NPY.array([[1, 1, 1], [1, 5, 7]]),
                      NPY.array([1, 2, 3]))

a.base_write('Base#3', 3, 3)
a.zone_write(3, 'Zone 01', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(3, 'Zone 02', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.zone_write(3, 'Zone 03', NPY.array([[3, 5, 7], [2, 4, 6], [0, 0, 0]]), CK.Structured)
a.coord_write(3, 1, CK.RealDouble, CK.CoordinateX_s, c01[0])
a.coord_write(3, 1, CK.RealDouble, CK.CoordinateY_s, c01[1])
a.coord_write(3, 1, CK.RealDouble, CK.CoordinateZ_s, c01[2])
a.coord_write(3, 2, CK.RealDouble, CK.CoordinateX_s, c02[0])
a.coord_write(3, 2, CK.RealDouble, CK.CoordinateY_s, c02[1])
a.coord_write(3, 2, CK.RealDouble, CK.CoordinateZ_s, c02[2])
a.coord_write(3, 3, CK.RealDouble, CK.CoordinateX_s, c03[0])
a.coord_write(3, 3, CK.RealDouble, CK.CoordinateY_s, c03[1])
a.coord_write(3, 3, CK.RealDouble, CK.CoordinateZ_s, c03[2])
a.boco_write(3, 1, 'BC 01', 12, 4, 2, NPY.array([[1, 1, 1], [3, 5, 1]]))
a.boco_write(3, 1, 'BC 02', 12, 4, 2, NPY.array([[1, 1, 7], [3, 5, 7]]))
t = a.conn_1to1_write(3, 1, 'Z-Conn#03>??:01', 'Zone 01',
                      NPY.array([[3, 1, 1], [3, 5, 7]]),
                      NPY.array([[1, 1, 1], [1, 5, 7]]),
                      NPY.array([1, 2, 3]))

a.close()

# ------------------------------------------------------------------------

a = Mll.pyCGNS('tmp/testmll156.hdf', Mll.MODE_READ)
print(a.conn_1to1_read(1, 1, 1))
print(a.conn_1to1_read(2, 1, 1))
print(a.conn_1to1_read(3, 1, 1))
a.close()


# ---
