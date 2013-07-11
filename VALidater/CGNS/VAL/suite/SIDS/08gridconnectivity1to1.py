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
tag='gridconnectivity1to1'
diag=True
vertexsize = 20
cellsize   = 7
ntris      = 12
def makeCorrectTree():
  T=CGL.newCGNSTree()
  b=CGL.newBase(T,'Base',3,3)
  z1=CGL.newZone(b,'Zone1',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
  g=CGL.newGridCoordinates(z1,'GridCoordinates')
  d=CGL.newDataArray(g,CGK.CoordinateX_s,NPY.ones((5,7,9),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateY_s,NPY.ones((5,7,9),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateZ_s,NPY.ones((5,7,9),dtype='d',order='F'))
  s=NPY.array([[vertexsize,cellsize,0]],dtype='i',order='F')
  z2=CGL.newZone(b,'Zone2',s,CGK.Unstructured_s)
  g=CGL.newGridCoordinates(z2,'GridCoordinates')
  d=CGL.newDataArray(g,CGK.CoordinateX_s,NPY.ones((vertexsize),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateY_s,NPY.ones((vertexsize),dtype='d',order='F'))
  d=CGL.newDataArray(g,CGK.CoordinateZ_s,NPY.ones((vertexsize),dtype='d',order='F'))
  tetras=CGL.newElements(z2,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
  tris=CGL.newElements(z2,'TRIS',CGK.TRI_3_s,NPY.ones((ntris*3),dtype='i'),NPY.array([[cellsize+1,cellsize+ntris]],'i',order='F'))
  z3=CGU.copyNode(z1,'Zone3')
  b[2].append(z3)
  z4=CGU.copyNode(z2,'Zone4')
  b[2].append(z4)
  z=[z1,z2,z3,z4]
  return (T,b,z)
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
zgc=CGL.newZoneGridConnectivity(z[2])
gc=CGL.newGridConnectivity1to1(zgc,'join3_1','Zone1',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 bad datatype'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
gc[1]=NPY.array([1],order='F')
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 absent opposite GridConnectivity'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 unvalid opposite GridConnectivity'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
zgc=CGL.newZoneGridConnectivity(z[2])
gc=CGL.newGridConnectivity1to1(zgc,'join3_1','Zone1',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
CGU.removeChildByName(gc,CGK.PointRangeDonor_s)
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 bad shape on Transform'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[+1,+2,+3]]))
zgc=CGL.newZoneGridConnectivity(z[2])
gc=CGL.newGridConnectivity1to1(zgc,'join3_1','Zone1',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 bad Transform'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,-4]))
zgc=CGL.newZoneGridConnectivity(z[2])
gc=CGL.newGridConnectivity1to1(zgc,'join3_1','Zone1',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+4]))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='gridconnectivity1to1 several 0 in Transform'
diag=False
(T,b,z)=makeCorrectTree()
zgc=CGL.newZoneGridConnectivity(z[0])
gc=CGL.newGridConnectivity1to1(zgc,'join1_3','Zone3',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,0,0]))
zgc=CGL.newZoneGridConnectivity(z[2])
gc=CGL.newGridConnectivity1to1(zgc,'join3_1','Zone1',NPY.array([[1,5],[1,7],[1,9]]),NPY.array([[1,5],[1,7],[1,9]]),NPY.array([+1,+2,+3]))
TESTS.append((tag,T,diag))
