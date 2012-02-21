#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils    as CGU
import CGNS.APP.probe.arrayutils as CGA

import numpy as NPY

cimport cython
cimport cpython
cimport numpy as CNPY

import vtk

# ----------------------------------------------------------------------------

class CGNSparser:
  
  def __init__(self,T):    
    self._zones={}
    self._zones_ns={}
    self._tree=T
    
  def parseZones(self):
    T=self._tree
    if (T[0]==None):
      T[0]=CGK.CGNSTree_s
      T[3]=CGK.CGNSTree_ts   
    for z in CGU.getAllNodesByTypeSet(T,[CGK.Zone_ts]):
      zT=CGU.nodeByPath(z,T)
      meshpath=z#CGU.removeFirstPathItem(z)
      g=CGU.getAllNodesByTypeSet(zT,[CGK.GridCoordinates_ts])[0]
      gT=CGU.nodeByPath(g,zT)
      cx=CGU.nodeByPath("%s/CoordinateX"%gT[0],gT)
      cy=CGU.nodeByPath("%s/CoordinateY"%gT[0],gT)
      cz=CGU.nodeByPath("%s/CoordinateZ"%gT[0],gT)
      zonetype=CGU.getAllNodesByTypeSet(zT,[CGK.ZoneType_ts])
      ztype=CGU.nodeByPath(zonetype[0],zT)      
      if (ztype[1].tostring()==CGK.Structured_s):          
        shx=cx[1].shape
        scx=cx[1].reshape(shx)
        scy=cy[1].reshape(shx)
        scz=cz[1].reshape(shx)        
        simin=[scx[0,:,:], scy[0,:,:], scz[0,:,:]]
        simax=[scx[-1,:,:],scy[-1,:,:],scz[-1,:,:]]
        sjmin=[scx[:,0,:], scy[:,0,:], scz[:,0,:]]
        sjmax=[scx[:,-1,:],scy[:,-1:], scz[:,-1,:]]
        skmin=[scx[:,:,0], scy[:,:,0], scz[:,:,0]]
        skmax=[scx[:,:,-1],scy[:,:,-1],scz[:,:,-1]]
        zp=z#CGU.removeFirstPathItem(z)        
        surfpaths=[zp+' {imin}',zp+' {imax}',
                   zp+' {jmin}',zp+' {jmax}',
                   zp+' {kmin}',zp+' {kmax}']
        bndlist=[]
        bcpaths=[]
        for nzbc in CGU.getAllNodesByTypeSet(zT,[CGK.ZoneBC_ts]):
          zbcT=CGU.nodeByPath(nzbc,zT)
          for nbc in CGU.getAllNodesByTypeSet(zbcT,[CGK.BC_ts]):
            bcpaths+=['%s/ZoneBC/%s'%(zp,nbc.split('/')[1])]
            bcT=CGU.nodeByPath(nbc,zbcT)
            for rbc in CGU.getAllNodesByTypeSet(bcT,[CGK.IndexRange_ts]):
              ptr=CGU.nodeByPath(rbc,bcT)[1].T.flat
              brg=[scx[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],
                   scy[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],
                   scz[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]]]
              bndlist+=[brg]
        self._zones[z]=([cx[1],cy[1],cz[1]],
                        [simin,simax,sjmin,sjmax,skmin,skmax],bndlist,
                        meshpath,surfpaths,bcpaths)
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
          if (et in sp.QUAD_SURFACE):
            pth=CGU.getPathAncestor(meshpath)+e+' {QUAD}'
            sl.append(list(sp.extQuadFacesPoints(ea,et,sn,mr,eb))+[pth])
          if (et in sp.TRI_SURFACE):
            pth=e+' {TRI}'
            sl.append(list(sp.extTriFacesPoints(ea,et,sn,mr,eb))+[pth])
        self._zones_ns[z]=([cx[1],cy[1],cz[1]],meshpath,et,sl)
    return None

#----------------------------------------------------------------------------
class Mesh(CGNSparser):

  def __init__(self,T):
    
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
                   CGK.PENTA_6: (vtk.vtkPolyhedron,(6,6)),
                   CGK.PENTA_15:(vtk.vtkPolyhedron,(6,15)),
                   CGK.PENTA_18:(vtk.vtkPolyhedron,(6,18)),
                   CGK.HEXA_8:  (vtk.vtkHexahedron,(8,8)),
                   CGK.HEXA_20: (vtk.vtkHexahedron,(8,20)),
                   CGK.HEXA_27: (vtk.vtkHexahedron,(8,27))}
    self.parseZones()

  def createActors(self):
    for z in self._zones.values():      
      self.do_vtk(z)
    self._actors+=self.createActors_ns()
    return self._actors

  def createActors_ns(self):
    actors=self.do_surface_ns(self._zones_ns)
    return actors
    
  def getObjectList(self):    
    return self._actors                 

  def getPathList(self):
    return [a[3] for a in self._actors]

  def getPathFromObject(self,selectedobject):
    for (o,p) in [(a[2],a[3]) for a in self._actors]:                  
        if (selectedobject==o): return p
    return ''
    
  def getObjectFromPath(self,selectedpath):
    for (o,p) in [(a[2],a[3]) for a in self._actors]:                  
        if (selectedpath==p): return o
    return ''
    
#  @cython.boundscheck(False)
  def do_volume(self,path,dx,dy,dz):
    cdef int p, i, j, k, idim, jdim, kdim
    cdef double x,y,z
    idim = dx.shape[0]
    jdim = dx.shape[1]
    kdim = dx.shape[2]
    pts=vtk.vtkPoints()
    pts.SetNumberOfPoints(idim*jdim*kdim)
    for k in range(kdim):
     for j in range(jdim):
      for i in range(idim):
       p=i+j*idim+k*idim*jdim
       x = (<double*>CNPY.PyArray_GETPTR1(dx,p))[0]
       y = (<double*>CNPY.PyArray_GETPTR1(dy,p))[0]
       z = (<double*>CNPY.PyArray_GETPTR1(dz,p))[0]
       pts.InsertPoint(p,x,y,z)
    g=vtk.vtkStructuredGrid()
    g.SetPoints(pts)
    g.SetExtent(0,idim-1,0,jdim-1,0,kdim-1)
    d=vtk.vtkDataSetMapper()
    d.SetInput(g)
    a=vtk.vtkActor()
    a.SetMapper(d)
    a.GetProperty().SetRepresentationToWireframe()
    return (a,a.GetBounds(),g,path)

#  @cython.boundscheck(False)
  def do_surface(self,surf,path):
    cdef int i, j, imax, jmax, p1, p2, p3, p4
    imax=surf[0].shape[0]
    jmax=surf[0].shape[1]
    tx=surf[0].flat
    ty=surf[1].flat
    tz=surf[2].flat
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
    am.SetInput(sg)
    a = vtk.vtkActor()
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    return (a,None,sg,path)

  def do_boundaries(self,bnd,path):
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
    am.SetInput(sg)
    a = vtk.vtkActor()
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    return (a,None,sg,path)

  def do_vtk(self,z):
      self._actors+=[self.do_volume(z[3],z[0][0],z[0][1],z[0][2])]
      for (s,sp) in zip(z[1],z[4]):
        self._actors+=[self.do_surface(s,sp)]
      for (b,sb) in zip(z[2],z[5]):
        self._actors+=[self.do_boundaries(b,sb)]
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
        am.SetInput(sg)
        a = vtk.vtkActor()
        a.SetMapper(am)
        a.GetProperty().SetRepresentationToWireframe()
        actors+=[(a,a.GetBounds(),sg,path)]
    return actors

  def def_volume(self,n):
    if self._vtkelts.has_key(n):
      return self._vtkelts[n]
    return None

#----------------------------------------------------------------------------

