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
tag='bc'
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
  tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
  zbc=CGL.newZoneBC(z)
  n=CGL.newBoundary(zbc,'BC',[range(1,cellsize+1)],btype=CGK.Null_s,family=None,pttype=CGK.PointList_s)
  g=CGL.newGridLocation(n,value=CGK.CellCenter_s)  
  return (T,b,z,zbc,n,g)
vertexsize = 20
cellsize   = 7
(T,b,z,zbc,n,g)=makeCorrectTree(vertexsize,cellsize)
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='bc both PointList and PointRange'
diag=False
(T,b,z,zbc,n,g)=makeCorrectTree(vertexsize,cellsize)
CGL.newPointRange(n,value=NPY.array([[1,cellsize]],'i'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='bc no PointList or PointRange'
diag=False
(T,b,z,zbc,n,g)=makeCorrectTree(vertexsize,cellsize)
CGU.removeChildByName(n,CGK.PointList_s)
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='bc FamilySpecified but not FamilyName'
diag=False
(T,b,z,zbc,n,g)=makeCorrectTree(vertexsize,cellsize)
n[1]=CGU.setStringAsArray(CGK.FamilySpecified_s)
TESTS.append((tag,T,diag))