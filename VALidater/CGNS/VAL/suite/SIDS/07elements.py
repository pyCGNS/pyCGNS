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

