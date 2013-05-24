#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
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
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((4*4),dtype='i'),NPY.array([[1,4]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad dims'
diag=False
T=CGL.newCGNSTree()
b=CGL.newBase(T,'{Base}',3,3)
z=CGL.newZone(b,'{Zone}',NPY.array([[4,5,0]],order='F'))
zt=CGU.hasChildName(z,CGK.ZoneType_s)
zt[1]=CGU.setStringAsArray(CGK.Unstructured_s)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((5*4),dtype='i'),NPY.array([[1,5]],'i',order='F'))
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
tag='zone unstructured correct ElementRange combination'
diag=True
(T,b,z)=makeUnstTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((3*4),dtype='i'),NPY.array([[1,3]],'i',order='F'))
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((4*8),dtype='i'),NPY.array([[4,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+6]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad ElementRange combination #1'
diag=False
(T,b,z)=makeUnstTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((3*4),dtype='i'),NPY.array([[1,3]],'i',order='F'))
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((5*8),dtype='i'),NPY.array([[3,7]],'i',order='F')) # Bad combination of ElementRange
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad ElementRange combination #2'
diag=False
(T,b,z)=makeUnstTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1+1,cellsize+7+1]],'i',order='F')) # should be cellsize+1,cellsize+7
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured bad ElementRange combination #3'
diag=False
(T,b,z)=makeUnstTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize,cellsize+7-1]],'i',order='F')) # should be cellsize+1,cellsize+7
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='zone unstructured NFACE without NGON'
diag=False
(T,b,z)=makeUnstTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+5]],'i',order='F'))
TESTS.append((tag,T,diag))