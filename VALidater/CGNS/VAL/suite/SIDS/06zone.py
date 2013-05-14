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
tag='zone structured'
diag=True
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone bad zonetype'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
zt=CGU.hasChildName(z,CGK.ZoneType_s)
zt[1]=CGU.setStringAsArray('Untruscutred')
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone bad zone size'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
zt=CGU.hasChildName(z,CGK.ZoneType_s)
zt[1]=CGU.setStringAsArray(CGK.Unstructured_s)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((4*4),dtype='i'),NPY.array([[1,4]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad dims'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[4,5,0]],order='F'))
zt=CGU.hasChildName(z,CGK.ZoneType_s)
zt[1]=CGU.setStringAsArray(CGK.Unstructured_s)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((5*4),dtype='i'),NPY.array([[1,5]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone structured bad dims'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,4,0],[7,7,0],[9,8,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone structured bad numpy array order'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[5,7,8],[4,6,8],[0,0,0]],order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured no elements'
diag=False
def makeUnstTree(vertexsize,cellsize):
  T=CGL.newCGNSTree()
  b=CGL.newBase(T,'Base',3,3)
  s=NPY.array([[vertexsize,cellsize,0]],dtype='i',order='F')
  z=CGL.newZone(b,'Zone',s,CGK.Unstructured_s)
  return (T,b,z)
vertexsize = 20
cellsize   = 7
(T,b,z)=makeUnstTree(vertexsize,cellsize)
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad ElementRange combination'
diag=False
(T,b,z)=makeUnstTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((3*4),dtype='i'),NPY.array([[1,3]],'i',order='F'))
tris=CGL.newElements(z,'TRIS',CGK.TRI_3_s,NPY.ones((5*3),dtype='i'),NPY.array([[3,7]],'i',order='F')) # Bad combination of ElementRange
TESTS.append((tag,T,diag))
