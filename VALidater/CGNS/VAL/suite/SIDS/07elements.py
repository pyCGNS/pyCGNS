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
tag='elements'
diag=True
def makeCorrectTree(vertexsize,cellsize):
  T=CGL.newCGNSTree()
  b=CGL.newBase(T,'Base',3,3)
  s=NPY.array([[vertexsize,cellsize,0]],dtype='i',order='F')
  z=CGL.newZone(b,'Zone',s,CGK.Unstructured_s)
  g=CGL.newGridCoordinates(z,'GridCoordinates')
  d=CGL.newDataArray(g,CGK.CoordinateX_s,NPY.ones((vertexsize),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateY_s,NPY.ones((vertexsize),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateZ_s,NPY.ones((vertexsize),dtype='d',order='F'))
  return (T,b,z)
vertexsize = 20
cellsize   = 7
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements absent children'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
CGU.removeChildByName(quads,CGK.ElementRange_s) # ElementRange child absent
CGU.removeChildByName(quads,CGK.ElementConnectivity_s) # ElementConnectivity child absent
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements out of range'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
quads[2][0][1][0]=vertexsize+1 # ElementConnectity element out of range
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements bad child shape'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones(((cellsize-1)*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F')) # bad ElementConnectivity node shape
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements correct NGON shape'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements uncorrect NGON shape'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements correct MIXED shape'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
element=CGL.newElements(z,'MIXED',CGK.MIXED_s,
                        NPY.array([CGK.ElementType[CGK.QUAD_4_s],9,9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.TRI_3_s],9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.TRI_3_s],9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.QUAD_4_s],9,9,9,9],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements uncorrect MIXED shape #1'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
element=CGL.newElements(z,'MIXED',CGK.MIXED_s,
                        NPY.array([CGK.ElementType[CGK.QUAD_4_s],9,9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.TRI_3_s],9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.QUAD_4_s],9,9,9,9],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements uncorrect MIXED shape #2'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
element=CGL.newElements(z,'MIXED',CGK.MIXED_s,
                        NPY.array([CGK.ElementType[CGK.QUAD_4_s],9,9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,999,
                                   CGK.ElementType[CGK.TRI_3_s],9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.QUAD_4_s],9,9,9,9],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements uncorrect element within MIXED'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
element=CGL.newElements(z,'MIXED',CGK.MIXED_s,
                        NPY.array([CGK.ElementType[CGK.QUAD_4_s],9,9,9,9,
                                   999,9,9,9,9,9,
                                   CGK.ElementType[CGK.TRI_3_s],9,9,9,
                                   CGK.ElementType[CGK.PYRA_5_s],9,9,9,9,9,
                                   CGK.ElementType[CGK.QUAD_4_s],9,9,9,9],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements FACE number of faces out of range'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
element=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,9,9,9,9,
                                   99999999,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))