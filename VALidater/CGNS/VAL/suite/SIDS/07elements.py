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
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements absent children'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
CGU.removeChildByName(tetras,CGK.ElementRange_s) # ElementRange child absent
CGU.removeChildByName(tetras,CGK.ElementConnectivity_s) # ElementConnectivity child absent
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements out of range'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
tetras[2][0][1][0]=vertexsize+1 # ElementConnectity element out of range
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements bad child shape'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones(((cellsize-1)*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F')) # bad ElementConnectivity node shape
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements correct NGON shape'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
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
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones((cellsize*4),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
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
tag='elements FACE'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
nface=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,1,2,-3,4,
                                   5,1,2,3,4,5,
                                   3,1,-2,3,
                                   5,1,2,3,4,-5,
                                   3,1,2,3,
                                   5,1,2,3,4,5,
                                   4,1,2,-3,4],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
ngon=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+5]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements FACE inconsistent dataarray'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
nface=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,1,2,-3,4,
                                   9999999,1,2,3,4,5,
                                   3,1,-2,3,
                                   5,1,2,3,4,-5,
                                   3,1,2,3,
                                   5,1,2,3,4,5,
                                   4,1,2,-3,4],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
ngon=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+5]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements FACE bad node shape'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
nface=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,1,2,-3,4,
                                   5,1,2,3,4,5,
                                   3,1,-2,3,
                                   5,1,2,3,4,-5,
                                   3,1,2,3,
                                   5,1,2,3,4,5,
                                   4,1,2,-3,4],dtype='i',order='F'),
                        NPY.array([[1,cellsize-1]],'i',order='F'))
ngon=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+5]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements FACE face index out of range'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
nface=CGL.newElements(z,'NFACE',CGK.NFACE_n_s,
                        NPY.array([4,1,2,-3,4,
                                   5,1,2,99999,4,5,
                                   3,1,-2,3,
                                   5,1,2,3,4,-5,
                                   3,1,2,3,
                                   5,1,2,3,4,5,
                                   4,1,2,-3,4],dtype='i',order='F'),
                        NPY.array([[1,cellsize]],'i',order='F'))
ngon=CGL.newElements(z,'NGON',CGK.NGON_n_s,
                        NPY.array([4,9,9,9,9,
                                   5,9,9,9,9,9,
                                   3,9,9,9,
                                   5,9,9,9,9,9,
                                   4,9,9,9,9],dtype='i',order='F'),
                        NPY.array([[cellsize+1,cellsize+5]],'i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements parentelements and parentelementsposition'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((cellsize*8),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
nquads=3
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((nquads*4),dtype='i'),NPY.array([[cellsize+1,cellsize+nquads]],'i',order='F'))
pe=CGL.newParentElements(quads,NPY.ones((nquads,2),dtype='i',order='F'))
pp=CGL.newParentElementsPosition(quads,NPY.ones((nquads,2),dtype='i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements bad parentelements shape and parentelementsposition datatype'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((cellsize*8),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
nquads=3
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((nquads*4),dtype='i'),NPY.array([[cellsize+1,cellsize+nquads]],'i',order='F'))
pe=CGL.newParentElements(quads,NPY.ones((nquads-1,2),dtype='i',order='F'))
pp=CGL.newParentElementsPosition(quads,NPY.ones((nquads,2),dtype='d',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements parentelements and parentelementsposition on bad element type (Warning)'
diag=True
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((cellsize*8),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
pe=CGL.newParentElements(tetras,NPY.ones((cellsize,2),dtype='i',order='F'))
pp=CGL.newParentElementsPosition(tetras,NPY.ones((cellsize,2),dtype='i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements parentelements bad values #1'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((cellsize*8),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
nquads=3
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((nquads*4),dtype='i'),NPY.array([[cellsize+1,cellsize+nquads]],'i',order='F'))
pe=CGL.newParentElements(quads,NPY.array([[1,2],[1,cellsize+1],[1,2]],dtype='i',order='F'))
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements parentelements bad values #2'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
nhexa=2
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((nhexa*8),dtype='i'),NPY.array([[1,nhexa]],'i',order='F'))
tetras=CGL.newElements(z,'TETRAS',CGK.TETRA_4_s,NPY.ones(((cellsize-nhexa)*4),dtype='i'),NPY.array([[nhexa+1,cellsize]],'i',order='F'))
nquads=3
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((nquads*4),dtype='i'),NPY.array([[cellsize+1,cellsize+nquads]],'i',order='F'))
pe=CGL.newParentElements(quads,NPY.ones((nquads,2),dtype='i',order='F'))
pe[1][0][1]=nhexa+1
TESTS.append((tag,T,diag))

#  -------------------------------------------------------------------------
tag='elements parentelementsposition bad face position'
diag=False
(T,b,z)=makeCorrectTree(vertexsize,cellsize)
hexas=CGL.newElements(z,'HEXAS',CGK.HEXA_8_s,NPY.ones((cellsize*8),dtype='i'),NPY.array([[1,cellsize]],'i',order='F'))
nquads=3
quads=CGL.newElements(z,'QUADS',CGK.QUAD_4_s,NPY.ones((nquads*4),dtype='i'),NPY.array([[cellsize+1,cellsize+nquads]],'i',order='F'))
pe=CGL.newParentElements(quads,NPY.ones((nquads,2),dtype='i',order='F'))
pp=CGL.newParentElementsPosition(quads,NPY.ones((nquads,2),dtype='i',order='F')*7) # 7 > 6 faces for hexas
TESTS.append((tag,T,diag))