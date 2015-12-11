#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from CGNS.NAV.moption import Q7OptionContext as OCTXT

import CGNS.PAT.cgnskeywords   as CGK
import CGNS.PAT.cgnsutils      as CGU
import CGNS.APP.lib.arrayutils as CGA

import CGNS.NAV.wmessages      as MSG

import numpy as NPY

cimport cython
cimport cpython
cimport numpy as CNPY

from PySide.QtCore import QCoreApplication

import vtk
import vtk.util.numpy_support as vtknpy

vers=vtk.vtkVersion()
VTK_VERSION_MINOR=vers.GetVTKMinorVersion()
VTK_VERSION_MAJOR=vers.GetVTKMajorVersion()

FAMILY_PATTERN=' :[%s]'

EXTRA_PATTERN=' :{%s}'
IMIN_PATTERN=EXTRA_PATTERN%'imin'
JMIN_PATTERN=EXTRA_PATTERN%'jmin'
KMIN_PATTERN=EXTRA_PATTERN%'kmin'
IMAX_PATTERN=EXTRA_PATTERN%'imax'
JMAX_PATTERN=EXTRA_PATTERN%'jmax'
KMAX_PATTERN=EXTRA_PATTERN%'kmax'

QUAD_PATTERN=EXTRA_PATTERN%'quad'
TRI_PATTERN =EXTRA_PATTERN%'tri ' # mandatory trailing space to have 4 chars

SIZE_PATTERN=len(TRI_PATTERN)
LIST_PATTERN=(IMIN_PATTERN,JMIN_PATTERN,KMIN_PATTERN,
              IMAX_PATTERN,JMAX_PATTERN,KMAX_PATTERN,
              QUAD_PATTERN,TRI_PATTERN)

# ----------------------------------------------------------------------------
class Q7vtkActor(vtk.vtkActor):
  __actortypes=['Zone','BC','Min/Max']
  def __init__(self,topo=None):
    if (topo in self.__actortypes):
        self.__type=topo
    else: self.__type='Zone'
  def topo(self):
    return self.__type

# ----------------------------------------------------------------------------
class ZoneData(object):
  __zones=[]
  def __init__(self,meshcoords,surfcoords,bndlist,path,surfpaths,bcpaths,sols,
               bcfams):
    self.meshcoordinates=meshcoords # 0
    self.surfcoordinates=surfcoords # 1
    self.bndlist=bndlist            # 2
    self.path=path                  # 3
    self.surfpaths=surfpaths        # 4
    self.bcpaths=bcpaths            # 5
    self.sols=sols                  # 6
    self.bcfams=bcfams
    self.X=self.meshcoordinates[0]
    self.Y=self.meshcoordinates[1]
    self.Z=self.meshcoordinates[2]
    self.surfaces=zip(self.surfpaths,self.surfcoordinates)
    self.boundaries=zip(self.bcpaths,self.bndlist,self.bcfams)
    ZoneData.__zones.append(self)
  @classmethod
  def zonelist(cls):
    for z in cls.__zones: yield z

class CGNSparser:
  
  def __init__(self,T):    
    self._zones={}
    self._zones_ns={}
    self._tree=T
    self._rsd=None
    
  def resfreshScreen():
    QCoreApplication.processEvents()

  def parseZones(self,zlist=[]):
    T=self._tree
    if (T[0]==None):
      T[0]=CGK.CGNSTree_s
      T[3]=CGK.CGNSTree_ts
    for r in CGU.getAllNodesByTypeSet(T,[CGK.ConvergenceHistory_ts]):
      self._rsd=CGU.nodeByPath(r,T)
    allfamilies=CGU.getAllFamilies(T)
    for z in CGU.getAllNodesByTypeSet(T,[CGK.Zone_ts]):
      zT=CGU.nodeByPath(z,T)
      if ((zlist==[]) or (z in zlist)):
        gnode=CGU.getAllNodesByTypeSet(zT,[CGK.GridCoordinates_ts])
        if (gnode==[]): return False
        for g in gnode:
          zg=g
          meshpath=zg
          gT=CGU.nodeByPath(g,zT)
          cx=CGU.nodeByPath("%s/CoordinateX"%gT[0],gT)
          cy=CGU.nodeByPath("%s/CoordinateY"%gT[0],gT)
          cz=CGU.nodeByPath("%s/CoordinateZ"%gT[0],gT)
          if ((cx is None) or (cy is None) or (cz is None)): return False
          if ((cx[1] is None) or (cy[1] is None) or (cz[1] is None)):
            return False
          zonetype=CGU.getAllNodesByTypeSet(zT,[CGK.ZoneType_ts])
          ztype=CGU.nodeByPath(zonetype[0],zT)
          if (ztype[1].tostring()==CGK.Structured_s):
            if (cx[1]==None) : return False
            acx=cx[1]
            acy=cy[1]
            acz=cz[1]
            shx=cx[1].shape
            scx=cx[1].reshape(shx)
            scy=cy[1].reshape(shx)
            scz=cz[1].reshape(shx)        
            simin=[scx[0,:,:], scy[0,:,:], scz[0,:,:]]
            simax=[scx[-1,:,:],scy[-1,:,:],scz[-1,:,:]]
            sjmin=[scx[:,0,:], scy[:,0,:], scz[:,0,:]]
            sjmax=[scx[:,-1,:],scy[:,-1,:], scz[:,-1,:]]
            skmin=[scx[:,:,0], scy[:,:,0], scz[:,:,0]]
            skmax=[scx[:,:,-1],scy[:,:,-1],scz[:,:,-1]]
            zp=zg
            surfpaths=[zp+IMIN_PATTERN,zp+IMAX_PATTERN,
                       zp+JMIN_PATTERN,zp+JMAX_PATTERN,
                       zp+KMIN_PATTERN,zp+KMAX_PATTERN]
            bndlist=[]
            bcpaths=[]
            bcfams=[]
            solutions=[]
            for sol in CGU.getAllNodesByTypeSet(zT,[CGK.FlowSolution_ts]):
              zsol=CGU.nodeByPath(sol,zT)
              for data in CGU.getAllNodesByTypeSet(zsol,[CGK.DataArray_ts]):
                dsol=CGU.nodeByPath(data,zsol)
                solutions+=[dsol]
            for nzbc in CGU.getAllNodesByTypeSet(zT,[CGK.ZoneBC_ts]):
              zbcT=CGU.nodeByPath(nzbc,zT)
              for nbc in CGU.getAllNodesByTypeSet(zbcT,[CGK.BC_ts]):
                bcpaths+=['%s/ZoneBC/%s'%(zp,nbc.split('/')[1])]
                bcnode=CGU.getNodeByPath(zbcT,nbc)
                lfbc =CGU.hasChildType(bcnode,CGK.FamilyName_ts)
                lfbc+=CGU.hasChildType(bcnode,CGK.AdditionalFamilyName_ts)
                nflbcs=[]
                for nlfbc in lfbc:
                  nflbcs+=[nlfbc[1].tostring()]
                bcfams+=[nflbcs]
                bcT=CGU.nodeByPath(nbc,zbcT)
                for rbc in CGU.getAllNodesByTypeSet(bcT,[CGK.IndexRange_ts]):
                  ptr=CGU.nodeByPath(rbc,bcT)[1].T.flat
                  brg=[scx[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],
                       scy[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],
                       scz[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]]]
                  bndlist+=[brg]
            # self._zones[zg]=([acx,acy,acz],
            #                 [simin,simax,sjmin,sjmax,skmin,skmax],bndlist,
            #                 meshpath,surfpaths,bcpaths,solutions)
            self._zones[zg]=ZoneData([acx,acy,acz],
                            [simin,simax,sjmin,sjmax,skmin,skmax],bndlist,
                            meshpath,surfpaths,bcpaths,solutions,
                            bcfams)
          elif (ztype[1].tostring()==CGK.Unstructured_s):
            volume={}
            surface={}
            typeset=[CGK.Elements_ts]
            elist=CGU.getAllNodesByTypeSet(zT,typeset)
            sp=CGA.SectionParse()
            mr=1
            sn=0
            sl=[]
            for e in elist:
              sn+=1
              ne=CGU.getNodeByPath(zT,e)[1]
              et=ne[0]
              eb=ne[1]
              ea=CGU.getNodeByPath(zT,e+'/'+CGK.ElementConnectivity_s)[1]
              if ((ea is not None) and (et in sp.QUAD_SURFACE)):
                pth=CGU.getPathAncestor(meshpath)+e+QUAD_PATTERN
                sl.append(list(sp.extQuadFacesPoints(ea,et,sn,mr,eb))+[pth])
              if ((ea is not None) and (et in sp.TRI_SURFACE)):
                pth=e+TRI_PATTERN
                sl.append(list(sp.extTriFacesPoints(ea,et,sn,mr,eb))+[pth])
            self._zones_ns[z]=([cx[1],cy[1],cz[1]],meshpath,et,sl)
    return True

#----------------------------------------------------------------------------
class Mesh(CGNSparser):

  def __init__(self,T,zlist):

    CGNSparser.__init__(self,T)    
    self._color=(1,0,0)
    self._actors=[]
    self._vtkelts={CGK.TRI_3:   (vtk.vtkTriangle,  (3,3)),
                   CGK.TRI_6:   (vtk.vtkTriangle,  (3,6)),
                   CGK.QUAD_4:  (vtk.vtkQuad,      (4,4)),
                   CGK.QUAD_8:  (vtk.vtkQuad,      (4,8)),
                   CGK.QUAD_9:  (vtk.vtkQuad,      (4,9)),
                   CGK.TETRA_4: (vtk.vtkTetra,     (4,4)),
                   CGK.TETRA_10:(vtk.vtkTetra,     (4,10)),
                   CGK.PYRA_5:  (vtk.vtkPyramid,   (5,5)),
                   CGK.PYRA_14: (vtk.vtkPyramid,   (5,14)),
                   CGK.HEXA_8:  (vtk.vtkHexahedron,(8,8)),
                   CGK.HEXA_20: (vtk.vtkHexahedron,(8,20)),
                   CGK.HEXA_27: (vtk.vtkHexahedron,(8,27))}

    if (VTK_VERSION_MINOR>8):
      self._vtkelts[CGK.PENTA_6]=(vtk.vtkPolyhedron,(6,6))
      self._vtkelts[CGK.PENTA_15]=(vtk.vtkPolyhedron,(6,15))
      self._vtkelts[CGK.PENTA_18]=(vtk.vtkPolyhedron,(6,18))
   
    try:
      self._status=self.parseZones(zlist)
    except (ValueError,IndexError):
      self._status=False

  def getResidus(self):
    return self._rsd

  def createActors(self):
    for z in ZoneData.zonelist():      
      self.do_vtk(z)
      QCoreApplication.processEvents()
    self._actors+=self.createActors_ns()
    return self._actors

  def createActors_ns(self):
    actors=self.do_surface_ns(self._zones_ns)
    return actors
    
  def getObjectList(self):    
    return self._actors                 

  # self._actor := [ vtk.vtkActor, bbox, vtk.vtk<grid>, path, dims ]
  def getPathList(self,filter=[]):
    if (filter==[]):
        r=[a[3] for a in self._actors]
        r.sort()
        return r
    else:
      r=[]
      for a in self._actors:
        if (a[0].topo() in filter): r.append(a[3])
      r.sort()
      return r

  def getFamilyList(self):
    return ('A','B','C')

  def getPathFromObject(self,selectedobject):
    for (o,p) in [(a[2],a[3]) for a in self._actors]:                  
        if (selectedobject==o): return p
    return ''
    
  def getObjectFromPath(self,selectedpath):
    for (o,p) in [(a[2],a[3]) for a in self._actors]:                  
        if (selectedpath==p): return o
    return ''

  def getDimsFromObject(self,selectedobject):
    for (o,p) in [(a[2],a[4]) for a in self._actors]:
      if (selectedobject==o): return p
    return (1,None)
    
#  @cython.boundscheck(False)
  def do_volume(self,path,CNPY.ndarray dx,CNPY.ndarray dy,CNPY.ndarray dz,
                solution):
    data=vtk.vtkIntArray()
    data.SetNumberOfComponents(3)
    data.SetName("index volume")
    cdef int p, i, j, k, idim, jdim, kdim
    cdef float  xf,yf,zf
    cdef double xd,yd,zd
    idim = dx.shape[0]
    jdim = dx.shape[1]
    kdim = dx.shape[2]
    pts=vtk.vtkPoints()
    pts.SetNumberOfPoints(idim*jdim*kdim)
    #pts.setData
    if (dx.dtype==NPY.float32):
      for k in xrange(kdim):
       for j in xrange(jdim):
        for i in xrange(idim):
         data.InsertNextTuple3(i+1,j+1,k+1)
         p=i+j*idim+k*idim*jdim
         xf = (<float*>CNPY.PyArray_GETPTR1(dx,p))[0]
         yf = (<float*>CNPY.PyArray_GETPTR1(dy,p))[0]
         zf = (<float*>CNPY.PyArray_GETPTR1(dz,p))[0]
         pts.InsertPoint(p,xf,yf,zf)
    else:
      for k in xrange(kdim):
       for j in xrange(jdim):
        for i in xrange(idim):
         data.InsertNextTuple3(i+1,j+1,k+1)
         p=i+j*idim+k*idim*jdim
         xd=((<double*>dx.data)+p)[0]
         yd=((<double*>dy.data)+p)[0]
         zd=((<double*>dz.data)+p)[0]
         pts.InsertPoint(p,xd,yd,zd)
    g=vtk.vtkStructuredGrid()
    g.SetPoints(pts)
    g.SetExtent(0,idim-1,0,jdim-1,0,kdim-1)
    d=vtk.vtkDataSetMapper()
    if (VTK_VERSION_MAJOR<6): d.SetInput(g)
    else:                     d.SetInputData(g)
    a=Q7vtkActor('Zone')
    a.SetMapper(d)
    a.GetProperty().SetRepresentationToWireframe()
    g.GetPointData().AddArray(data)
    for s in solution:
      if ((s[1] is not None) and (s[1].shape==(idim-1,jdim-1,kdim-1))):
        array=vtk.vtkFloatArray()
        array.SetName(s[0])
        for k in xrange(kdim-1):          
          for j in xrange(jdim-1):
            for i in xrange(idim-1):
              array.InsertNextTuple1(s[1][i][j][k])
        g.GetCellData().AddArray(array)
    return (a,a.GetBounds(),g,path,(0,(idim,jdim,kdim)))

#  @cython.boundscheck(False)
  def do_surface_double_3d(self,path,surf):
    cdef int n, np, i, j, imax, jmax, p1, p2, p3, p4
    cdef double xs,ys,zs
    cdef CNPY.ndarray[CNPY.float64_t, ndim=2] _tx
    cdef CNPY.ndarray[CNPY.float64_t, ndim=2] _ty
    cdef CNPY.ndarray[CNPY.float64_t, ndim=2] _tz
    cdef double* tx
    cdef double* ty
    cdef double* tz
    imax=surf[0].shape[0]
    jmax=surf[0].shape[1]
    _tx=surf[0]
    _ty=surf[1]
    _tz=surf[2]
    tx=<double*>_tx.data
    ty=<double*>_ty.data
    tz=<double*>_tz.data
    sg=vtk.vtkUnstructuredGrid()
    sg.Allocate(1, 1)
    n=0
    qp = vtk.vtkPoints() # TODO: add family label as attribute
    for j in xrange(jmax-1):
     for i in xrange(imax-1):
      p1=j+      i*jmax +0
      p2=j+      i*jmax +1
      p3=j+jmax+ i*jmax +1
      p4=j+jmax+ i*jmax +0
      aq=vtk.vtkQuad()
      aqp=aq.GetPointIds()
      np=n*4
      xs=tx[p1]
      ys=ty[p1]
      zs=tz[p1]
      qp.InsertPoint(np,xs,ys,zs)
      aqp.SetId(0,np)
      np+=1
      qp.InsertPoint(np,tx[p2],ty[p2],tz[p2])
      aqp.SetId(1,np)
      np+=1
      qp.InsertPoint(np,tx[p3],ty[p3],tz[p3])
      aqp.SetId(2,np)
      np+=1
      qp.InsertPoint(np,tx[p4],ty[p4],tz[p4])
      aqp.SetId(3,np)
      sg.InsertNextCell(aq.GetCellType(), aqp)
      n+=1
    qp=vtk.vtkPoints()
    sg.SetPoints(qp)
    am = vtk.vtkDataSetMapper()
    if (VTK_VERSION_MAJOR<6): am.SetInput(sg)
    else:                     am.SetInputData(sg)
    a = Q7vtkActor('Min/Max')
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    return (a,None,sg,path,(1,None),None)

  def do_boundaries(self,path,bnd,fams):
    cdef int i, j, imax, jmax, p1, p2, p3, p4
    max=[x for x in bnd[0].shape if x!=1]
    imax=max[0]
    jmax=max[1]
    tx=bnd[0].flat
    ty=bnd[1].flat
    tz=bnd[2].flat
    sg=vtk.vtkUnstructuredGrid()
    sg.Allocate(1, 1)
    n=0
    qp = vtk.vtkPoints()
    for j in range(jmax-1):
     for i in range(imax-1):
      p1=j+      i*jmax +0
      p2=j+      i*jmax +1
      p3=j+jmax+ i*jmax +1
      p4=j+jmax+ i*jmax +0
      qp.InsertPoint(n*4+0,tx[p1],ty[p1],tz[p1])
      qp.InsertPoint(n*4+1,tx[p2],ty[p2],tz[p2])
      qp.InsertPoint(n*4+2,tx[p3],ty[p3],tz[p3])
      qp.InsertPoint(n*4+3,tx[p4],ty[p4],tz[p4])
      aq = vtk.vtkQuad()
      aq.GetPointIds().SetId(0, n*4+0)
      aq.GetPointIds().SetId(1, n*4+1)
      aq.GetPointIds().SetId(2, n*4+2)
      aq.GetPointIds().SetId(3, n*4+3)
      sg.InsertNextCell(aq.GetCellType(), aq.GetPointIds())
      n+=1
    sg.SetPoints(qp)
    am = vtk.vtkDataSetMapper()
    if (VTK_VERSION_MAJOR<6): am.SetInput(sg)
    else:                     am.SetInputData(sg)
    a = Q7vtkActor('BC')
    a.SetMapper(am) 
    a.GetProperty().SetRepresentationToWireframe()
    return (a,None,sg,path,(1,None),None)

  def do_vtk(self,z):
      self._actors+=[self.do_volume(z.path,z.X,z.Y,z.Z,z.sols)]
      return
      for (path,coords) in z.surfaces:
        self._actors+=[self.do_surface_double_3d(path,coords)]
      for (path,coords,fams) in z.boundaries:
        self._actors+=[self.do_boundaries(path,coords,fams)]
      return

#  @cython.boundscheck(False)
  def do_surface_ns(self,zones):
    cdef int e,elts,n,npe,idg,idx
    actors=[]
    for zn in zones:
      dx=zones[zn][0][0]
      dy=zones[zn][0][1]
      dz=zones[zn][0][2]
      for surf in zones[zn][-1]:
        path=surf[-1]
        if (surf[0]==CGK.QUAD_4): npe=4
        else: npe=3
        sf=surf[1]
        elts=len(sf)/npe
        sg=vtk.vtkUnstructuredGrid()
        sg.Allocate(1, 1)
        e=0
        qp=vtk.vtkPoints()
        vtkelt=self.def_volume(surf[0])[0]
        while (e<elts):
          n=0
          aq=vtkelt()
          while (n<npe):
            idg=(e*npe+n)
            ids=sf[idg]-1
            qp.InsertPoint(idg,dx[ids],dy[ids],dz[ids])
            aq.GetPointIds().SetId(n,idg)
            n+=1
          sg.InsertNextCell(aq.GetCellType(),aq.GetPointIds())
          e+=1
        sg.SetPoints(qp)    
        am = vtk.vtkDataSetMapper()
        if (VTK_VERSION_MAJOR<6): am.SetInput(sg)
        else:                     am.SetInputData(sg)
        a = Q7vtkActor()
        a.SetMapper(am)
        a.GetProperty().SetRepresentationToWireframe()
        actors+=[(a,a.GetBounds(),sg,path,(2,None))]
    return actors

  def def_volume(self,n):
    if self._vtkelts.has_key(n):
      return self._vtkelts[n]
    return None

#----------------------------------------------------------------------------

