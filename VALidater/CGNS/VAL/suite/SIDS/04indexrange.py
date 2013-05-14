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
tag='indexrange'
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
tag='indexrange index bad ordered' # this rises a warning, not an error
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[cellsize,1]],'i',order='F')) # element range not ordered
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='indexrange index out of range'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones(((cellsize+1)*4),dtype='i'),NPY.array([[1,cellsize+1]],'i',order='F')) # element index out of range
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='indexrange bad node shape'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([1,cellsize],'i',order='F')) # ElementRange bad node shape
TESTS.append((tag,T,diag))