#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY

TESTS=[]

#  -------------------------------------------------------------------------
tag='zone structured #1'
diag=True
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone structured bad dims #1'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,7,0],[9,8,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone structured bad ordering #1'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,7,8],[4,6,8],[0,0,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
