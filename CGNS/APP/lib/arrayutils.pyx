
# ----------------------------------------------------------------------
#
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import CGNS.PAT.cgnskeywords as CGK
import  numpy as  NPY
cimport numpy as  NPY
import random

NPY.import_array()

cimport CGNS.APP.lib.hashutils as HUT

# ----------------------------------------------------------------------
cdef int _trisFromTri(NPY.ndarray[NPY.int32_t, ndim=1] itris,
                      int itriscount,
                      NPY.ndarray[NPY.int32_t, ndim=1] otris,
                      int npe,int minr, int bnd,
                      HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=itriscount
  if (bnd): maxc=bnd
  r  =<int*>otris.data
  d  =<int*>itris.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+3],0,sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+3]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+3],0,minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _quadsFromQuad(NPY.ndarray[NPY.int32_t, ndim=1] iquads,
                        int iquadscount,
                        NPY.ndarray[NPY.int32_t, ndim=1] oquads,
                        int npe,int minr, int bnd,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=iquadscount
  if (bnd): maxc=bnd
  r  =<int*>oquads.data
  d  =<int*>iquads.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+3],d[dix+4],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+3]
      r[rix+3]=d[dix+4]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+3],d[dix+4],minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _trisFromTetra(NPY.ndarray[NPY.int32_t, ndim=1] tetras,
                        int tetrascount,
                        NPY.ndarray[NPY.int32_t, ndim=1] tris,
                        int npe,int minr, int bnd,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=tetrascount
  if (bnd): maxc=bnd
  r  =<int*>tris.data
  d  =<int*>tetras.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+3],d[dix+2],0,sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+2]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+3],d[dix+2],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+4],0,sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+4]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+4],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+4],0,sn)):
      r[rix+0]=d[dix+2]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+4]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+4],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+3],d[dix+1],d[dix+4],0,sn)):
      r[rix+0]=d[dix+3]
      r[rix+1]=d[dix+1]
      r[rix+2]=d[dix+4]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+3],d[dix+1],d[dix+4],0,minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _trisFromPyra(NPY.ndarray[NPY.int32_t, ndim=1] pyras,
                       int pyrascount,
                       NPY.ndarray[NPY.int32_t, ndim=1] tris,
                       int npe,int minr, int bnd,
                       HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,m=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  cdef int *pr
  cdef int *ps
  maxc=pyrascount
  if (bnd): maxc=bnd
  r  =<int*>tris.data
  d  =<int*>pyras.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+5],0,sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+5]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+5],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+5],0,sn)):
      r[rix+0]=d[dix+2]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+5]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+5],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+3],d[dix+4],d[dix+5],0,sn)):
      r[rix+0]=d[dix+3]
      r[rix+1]=d[dix+4]
      r[rix+2]=d[dix+5]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+3],d[dix+4],d[dix+5],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+4],d[dix+1],d[dix+5],0,sn)):
      r[rix+0]=d[dix+4]
      r[rix+1]=d[dix+1]
      r[rix+2]=d[dix+5]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+4],d[dix+1],d[dix+5],0,minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _quadsFromPyra(NPY.ndarray[NPY.int32_t, ndim=1] pyras,
                        int pyrascount,
                        NPY.ndarray[NPY.int32_t, ndim=1] quads,
                        int npe,int minr, int bnd,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=pyrascount
  if (bnd): maxc=bnd
  r  =<int*>quads.data
  d  =<int*>pyras.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+4],d[dix+3],d[dix+2],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+4]
      r[rix+2]=d[dix+3]
      r[rix+3]=d[dix+2]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+4],d[dix+3],d[dix+2],minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _trisFromPenta(NPY.ndarray[NPY.int32_t, ndim=1] pentas,
                        int pentascount,
                        NPY.ndarray[NPY.int32_t, ndim=1] tris,
                        int npe,int minr, int bnd,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,m=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  cdef int *pr
  cdef int *ps
  maxc=pentascount
  if (bnd): maxc=bnd
  r  =<int*>tris.data
  d  =<int*>pentas.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+3],d[dix+2],0,sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+2]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+3],d[dix+2],0,minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+4],d[dix+5],d[dix+6],0,sn)):
      r[rix+0]=d[dix+4]
      r[rix+1]=d[dix+5]
      r[rix+2]=d[dix+6]
      rix=rix+3
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+4],d[dix+5],d[dix+6],0,minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _quadsFromPenta(NPY.ndarray[NPY.int32_t, ndim=1] pentas,
                         int pentascount,
                         NPY.ndarray[NPY.int32_t, ndim=1] quads,
                         int npe,int minr, int bnd,
                         HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=pentascount
  if (bnd): maxc=bnd
  r  =<int*>quads.data
  d  =<int*>pentas.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+5],d[dix+4],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+5]
      r[rix+3]=d[dix+4]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+5],d[dix+4],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+6],d[dix+5],sn)):
      r[rix+0]=d[dix+2]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+6]
      r[rix+3]=d[dix+5]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+6],d[dix+5],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+3],d[dix+1],d[dix+4],d[dix+6],sn)):
      r[rix+0]=d[dix+3]
      r[rix+1]=d[dix+1]
      r[rix+2]=d[dix+4]
      r[rix+3]=d[dix+6]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+3],d[dix+1],d[dix+4],d[dix+6],minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _quadsFromHexa(NPY.ndarray[NPY.int32_t, ndim=1] hexas,
                        int hexascount,
                        NPY.ndarray[NPY.int32_t, ndim=1] quads,
                        int npe,int minr, int bnd,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,m=0,f=0,dix,maxc
  cdef int *d
  cdef int *r
  maxc=hexascount
  if (bnd): maxc=bnd
  r  =<int*>quads.data
  d  =<int*>hexas.data
  while (p<maxc):
    dix=p*npe-1
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+4],d[dix+3],d[dix+2],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+4]
      r[rix+2]=d[dix+3]
      r[rix+3]=d[dix+2]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+4],d[dix+3],d[dix+2],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+6],d[dix+5],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+2]
      r[rix+2]=d[dix+6]
      r[rix+3]=d[dix+5]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+2],d[dix+6],d[dix+5],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+7],d[dix+6],sn)):
      r[rix+0]=d[dix+2]
      r[rix+1]=d[dix+3]
      r[rix+2]=d[dix+7]
      r[rix+3]=d[dix+6]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+2],d[dix+3],d[dix+7],d[dix+6],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+3],d[dix+4],d[dix+8],d[dix+7],sn)):
      r[rix+0]=d[dix+3]
      r[rix+1]=d[dix+4]
      r[rix+2]=d[dix+8]
      r[rix+3]=d[dix+7]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+3],d[dix+4],d[dix+8],d[dix+7],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+1],d[dix+5],d[dix+8],d[dix+4],sn)):
      r[rix+0]=d[dix+1]
      r[rix+1]=d[dix+5]
      r[rix+2]=d[dix+8]
      r[rix+3]=d[dix+4]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+1],d[dix+5],d[dix+8],d[dix+4],minr+f,sn)
    if (not HUT.fetchHashEntry(h,hsz,d[dix+5],d[dix+6],d[dix+7],d[dix+8],sn)):
      r[rix+0]=d[dix+5]
      r[rix+1]=d[dix+6]
      r[rix+2]=d[dix+7]
      r[rix+3]=d[dix+8]
      rix=rix+4
      f+=1
      HUT.addHashEntry(h,hsz,d[dix+5],d[dix+6],d[dix+7],d[dix+8],minr+f,sn)
    p+=1
  return f

# ----------------------------------------------------------------------
cdef int _exteriorFaces(NPY.ndarray[NPY.int32_t, ndim=1] elts,
                        int eltscount,
                        NPY.ndarray[NPY.int32_t, ndim=1] xfaces,
                        int npe,
                        HUT.F_entry_ptr h,int hsz,int sn):
  cdef int p=0,rix=0,f=0,dix,dp=0
  cdef int *d
  cdef int *r
  r  =<int*>xfaces.data
  d  =<int*>elts.data
  while (p<eltscount):
    dix=p*npe
    if (npe==4): dp=d[dix+3]
    if (HUT.exteriorHashEntry(h,hsz,d[dix+0],d[dix+1],d[dix+2],dp,sn)==1):
      r[rix+0]=d[dix+0]
      r[rix+1]=d[dix+1]
      r[rix+2]=d[dix+2]
      if (npe==4):
          r[rix+3]=dp
          rix=rix+4
      else:
          rix=rix+3
      f+=1
    p+=1
  return f

# ----------------------------------------------------------------------
cdef class SectionParse(object):
  cdef HUT.F_entry_ptr facesH
  cdef int sizeH
  QUAD_SURFACE=[CGK.QUAD_4,CGK.QUAD_8,CGK.QUAD_9,
                CGK.PENTA_6,CGK.PENTA_15,CGK.PENTA_18,
                CGK.HEXA_8,CGK.HEXA_20,CGK.HEXA_27,
                CGK.PYRA_5,CGK.PYRA_13,CGK.PYRA_14]

  TRI_SURFACE =[CGK.TRI_3,CGK.TRI_6,
                CGK.PENTA_6,CGK.PENTA_15,CGK.PENTA_18,
                CGK.TETRA_4,CGK.TETRA_10,
                CGK.PYRA_5,CGK.PYRA_13,CGK.PYRA_14]

  QUAD_TYPES=[CGK.QUAD_4,CGK.QUAD_8,CGK.QUAD_9]
  PENTA_TYPES=[CGK.PENTA_6,CGK.PENTA_15,CGK.PENTA_18]
  HEXA_TYPES=[CGK.HEXA_8,CGK.HEXA_20,CGK.HEXA_27]
  PYRA_TYPES=[CGK.PYRA_5,CGK.PYRA_13,CGK.PYRA_14]
  TRI_TYPES=[CGK.TRI_3,CGK.TRI_6]
  TETRA_TYPES=[CGK.TETRA_4,CGK.TETRA_10]
  
  SURFACE_TYPES=QUAD_TYPES+TRI_TYPES

  def __cinit__(self):
    self.sizeH=44497
    self.sizeH=1686049
    self.facesH=HUT.newHashTable(self.sizeH)

  def __dealloc__(self):
    HUT.freeHashTable(self.facesH,self.sizeH)

  def extQuadFacesPoints(self,ea,et,sn,mr,b):
    if (et in self.QUAD_TYPES):  qnpe=1
    if (et in self.PENTA_TYPES): qnpe=3
    if (et in self.HEXA_TYPES):  qnpe=6
    if (et in self.PYRA_TYPES):  qnpe=1
    enpe=CGK.ElementTypeNPE[CGK.ElementType_l[et]]
    fnpe=4
    f=0
    ez=len(ea.flat)
    ec=ez//enpe
    re=NPY.zeros((ec*qnpe*fnpe),dtype=NPY.int32)
    h=self.facesH
    hsz=self.sizeH
    if (et in self.QUAD_TYPES):
      re=ea
      f=ec
    if (et in self.PENTA_TYPES):f=_quadsFromPenta(ea,ec,re,enpe,mr,b,h,hsz,sn)
    if (et in self.HEXA_TYPES): f=_quadsFromHexa(ea,ec,re,enpe,mr,b,h,hsz,sn)
    if (et in self.PYRA_TYPES): f=_quadsFromPyra(ea,ec,re,enpe,mr,b,h,hsz,sn)
    rx=NPY.zeros((f*fnpe),dtype=NPY.int32)
    re=re[:f*fnpe].copy()
    if (et in self.SURFACE_TYPES):
      rx=re
      fx=f
    else:
      fx=_exteriorFaces(re[:f*fnpe].copy(),f,rx,fnpe,h,hsz,sn)
    return (CGK.QUAD_4,rx[:fx*fnpe],mr,f+mr,fx+mr)

  def extTriFacesPoints(self,ea,et,sn,mr,b):
    if (et in self.TRI_TYPES):   qnpe=1
    if (et in self.PENTA_TYPES): qnpe=2
    if (et in self.TETRA_TYPES): qnpe=4
    if (et in self.PYRA_TYPES):  qnpe=4
    enpe=CGK.ElementTypeNPE[CGK.ElementType_l[et]]
    fnpe=3
    f=0
    ez=len(ea.flat)
    ec=ez//enpe
    re=NPY.zeros((ec*qnpe*fnpe),dtype=NPY.int32)
    h=self.facesH
    hsz=self.sizeH
    if (et in self.TRI_TYPES):
      re=ea
      f=ec
    if (et in self.PENTA_TYPES):f=_trisFromPenta(ea,ec,re,enpe,mr,b,h,hsz,sn)
    if (et in self.TETRA_TYPES):f=_trisFromTetra(ea,ec,re,enpe,mr,b,h,hsz,sn)
    if (et in self.PYRA_TYPES): f=_trisFromPyra(ea,ec,re,enpe,mr,b,h,hsz,sn)
    rx=NPY.zeros((f*fnpe),dtype=NPY.int32)
    if (et in self.SURFACE_TYPES):
      rx=re
      fx=f
    else:
      fx=_exteriorFaces(re[:f*fnpe].copy(),f,rx,fnpe,h,hsz,sn)
    return (CGK.TRI_3,rx[:fx*fnpe].copy(),mr,f+mr,fx+mr)
    
# ----------------------------------------------------------------------
# --- last line
