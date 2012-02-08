#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils    as CGU
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
      meshpath=CGU.removeFirstPathItem(z)
      g=CGU.getAllNodesByTypeSet(zT,[CGK.GridCoordinates_ts])[0]
      gT=CGU.nodeByPath(g,zT)
      cx=CGU.nodeByPath("%s/CoordinateX"%gT[0],gT)
      cy=CGU.nodeByPath("%s/CoordinateY"%gT[0],gT)
      cz=CGU.nodeByPath("%s/CoordinateZ"%gT[0],gT)
      zonetype=CGU.getAllNodesByTypeSet(zT,[CGK.ZoneType_ts])
      type=CGU.nodeByPath(zonetype[0],zT)      
      if (type[1].tostring()==CGK.Structured_s):          
        shx=cx[1].shape
        scx=cz[1].reshape(shx)
        scy=cy[1].reshape(shx)
        scz=cx[1].reshape(shx)
        simin=[scx[0,:,:], scy[0,:,:], scz[0,:,:]]
        simax=[scx[-1,:,:],scy[-1,:,:],scz[-1,:,:]]
        sjmin=[scx[:,0,:], scy[:,0,:], scz[:,0,:]]
        sjmax=[scx[:,-1,:],scy[:,-1:], scz[:,-1,:]]
        skmin=[scx[:,:,0], scy[:,:,0], scz[:,:,0]]
        skmax=[scx[:,:,-1],scy[:,:,-1],scz[:,:,-1]]
        zp=CGU.removeFirstPathItem(z)
        surfpaths=[zp+'/[imin]',zp+'/[imax]',
                 zp+'/[jmin]',zp+'/[jmax]',
                 zp+'/[kmin]',zp+'/[kmax]']
        bndlist=[]
        bcpaths=[]
        for nzbc in CGU.getAllNodesByTypeSet(zT,[CGK.ZoneBC_ts]):
          zbcT=CGU.nodeByPath(nzbc,zT)
          for nbc in CGU.getAllNodesByTypeSet(zbcT,[CGK.BC_ts]):
            bcpaths+=['%s/[%s]'%(zp,nbc.split('/')[1])]
            bcT=CGU.nodeByPath(nbc,zbcT)
            for rbc in CGU.getAllNodesByTypeSet(bcT,[CGK.IndexRange_ts]):
              ptr=CGU.nodeByPath(rbc,bcT)[1].T.flat                                                                      
              brg=[scx[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],scy[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]],scz[ptr[0]-1:ptr[3],ptr[1]-1:ptr[4],ptr[2]-1:ptr[5]]]
              bndlist+=[brg]
        self._zones[z]=([cx[1],cy[1],cz[1]],[simin,simax,sjmin,sjmax,skmin,skmax],bndlist,
                      meshpath,surfpaths,bcpaths)
      elif (type[1].tostring()==CGK.Unstructured_s):
        volume={}
        surface={}
        for e in CGU.getAllNodesByTypeSet(zT,[CGK.Elements_ts]):
          zbcT=CGU.nodeByPath(e,zT)
          path=CGU.removeFirstPathItem(e)
          connectivity=CGU.getAllNodesByTypeSet(zbcT,[CGK.DataArray_ts])[0]
          rang=CGU.getAllNodesByTypeSet(zbcT,[CGK.IndexRange_ts])[0]
          co=CGU.nodeByPath(connectivity,zbcT)[1]   
          ra=CGU.nodeByPath(rang,zbcT)[1]
          if (zbcT[1][0]>=10):
            volume[e]=zbcT[1][0],co,ra
          else:
            surface[e]=zbcT[1][0],co,ra                   
        self._zones_ns[z]=([cx[1],cy[1],cz[1]],meshpath,volume,surface)
    return None

#----------------------------------------------------------------------------
 
class Mesh(CGNSparser):

  def __init__(self,T):
    CGNSparser.__init__(self,T)
    self._color=(1,0,0)
    self._actors=[]
    self._actors_ns=[]
    self.parseZones()
         
  def createActors(self):
    for z in self._zones.values():
      self.do_vtk(z)
    return self._actors

  def createActors_ns(self):
    actors=[]
    for k,n in self._zones_ns.iteritems():
      for p,m in n[3].iteritems():
        ti={}
        tx={}
        surface=self.parsesurface(m,ti,tx)
      for l,o in n[2].iteritems():
        volume=self.parsevolume(o,surface)
        for u,i in n[3].iteritems():
          index=self.extractFaces(volume[0],i)
          if (index!=[]):
            actors+=[self.do_surface_ns(index,n[0],i[0])]
    return actors
        
  def do_surface_ns(self,index,coordinates,shape):
    qp=vtk.vtkPoints()
    sg=vtk.vtkUnstructuredGrid()
    sg.Allocate(1, 1)
    pt=self.def_volume(shape)[1][0]
    for n in range(0,len(index),pt):
      aq=self.def_volume(shape)[0]
      for m in range(pt):
        qp.InsertPoint(n+m,coordinates[0][index[n+m]-1],coordinates[1][index[n+m]-1],coordinates[2][index[n+m]-1])
        aq.GetPointIds().SetId(m,n+m)
      sg.InsertNextCell(aq.GetCellType(),aq.GetPointIds())
    sg.SetPoints(qp)    
    am = vtk.vtkDataSetMapper()
    am.SetInput(sg)
    a = vtk.vtkActor()
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    return a
        
  def extractFaces(self,volume,k):
    pt=self.def_volume(k[0])[1][0]
    step=self.def_volume(k[0])[1][1]
    index=[]
    for i in volume:
      if ((i>=k[2][0]) and (i<=k[2][1])):
        n=i-k[2][0]
        for j in range(pt):
          index+=[k[1][n*step+j]]
    return index          

  def parsesurface(self,connectivity,ti,tx):
     face=connectivity[2][0]
     step=self.def_volume(connectivity[0])[1][1]
     for i in range(0,len(connectivity[1]),step):
       if ((connectivity[0]==5) or (connectivity[0]==6)):
         (ti,tx)=self.parsetri(ti,tx,face,connectivity[1],i)
       elif (connectivity[0]==7 or (connectivity[0]==8) or (connectivity[0]==9)):
         (ti,tx)=self.parsequad(ti,tx,face,connectivity[1],i)
       face+=1
     return tx

  def parsevolume(self,connectivity,surface):
    faces=[]
    if ((connectivity[0]==17) or (connectivity[0]==18) or (connectivity[0]==19)):
      faces+=[self.parsehex(surface,connectivity[1])]
    elif ((connectivity[0]==10) or (connectivity[0]==11)):
      faces+=[self.parsetetra(surface,connectivity[1])]                 
    elif ((connectivity[0]==12) or (connectivity[0]==13)):
      faces+=[self.parsepyra(surface,connectivity[1])]                    
    return faces
   
  def getObjectList(self):  
    return self._actors                 

  def getPathList(self):
    return [a[3] for a in self._actors]

  def getPathFromObject(self,selectedobject):
    for (o,p) in [(a[2],a[3]) for a in self._actors]:
        if (selectedobject==o): return p
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
    for i in range(idim):
     for j in range(jdim):
      for k in range(kdim):
       p=i+j*idim+k*idim*jdim
       x = (<double*>CNPY.PyArray_GETPTR1(dx,p))[0]
       y = (<double*>CNPY.PyArray_GETPTR1(dy,p))[0]
       z = (<double*>CNPY.PyArray_GETPTR1(dz,p))[0]
       pts.InsertPoint(k+j*kdim+i*kdim*jdim,z,y,x)
    g=vtk.vtkStructuredGrid()
    g.SetPoints(pts)
    g.SetExtent(0,kdim-1,0,jdim-1,0,idim-1)
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

  def sorting2(self,p1,p2):
     t=p1
     if p1>p2:
          p1=p2
          p2=t
     return p1,p2
     
  def sorting3(self,p1,p2,p3):
    (p1,p3)=self.sorting2(p1,p3)
    (p2,p3)=self.sorting2(p2,p3)
    return p1,p2,p3

  def sorting4(self,p1,p2,p3,p4):
    (p1,p4)=self.sorting2(p1,p4)
    (p2,p4)=self.sorting2(p2,p4)
    (p3,p4)=self.sorting2(p3,p4)
    return p1,p2,p3,p4
     
  def sort(self,p1,p2,p3,p4):
    (p1,p2)=self.sorting2(p1,p2)
    (p1,p2,p3)=self.sorting3(p1,p2,p3)
    (p1,p2,p3,p4)=self.sorting4(p1,p2,p3,p4)
    return p1,p2,p3,p4
                   
  def addFaceIntAndExt(self,ti,tx,face,p1,p2,p3,p4=0):
    (p1,p2,p3,p4)=self.sort(p1,p2,p3,p4)
    k="%12s%12s%12s%12s"%(p1,p2,p3,p4)
    if (tx.has_key(k)):
         ti[k]=face
         del tx[k]
    else: tx[k]=face
    return (ti,tx)

  def getFace(self,t,p1,p2,p3,p4=0):
    (t1,t2,t3,t4)=(p1,p2,p3,p4)
    (p1,p2,p3,p4)=self.sort(p1,p2,p3,p4)
    k="%12s%12s%12s%12s"%(p1,p2,p3,p4)
    if (t.has_key(k)): return t[k]
    return -1

  def parsetri(self,ti,tx,n,connectivity,i):
    (ti,tx)=self.addFaceIntAndExt(ti,tx,n,connectivity[i],connectivity[i+1],connectivity[i+2])
    return (ti,tx)

  def parsequad(self,ti,tx,n,connectivity,i):
     (ti,tx)=self.addFaceIntAndExt(ti,tx,n,connectivity[i],connectivity[i+1],connectivity[i+2],connectivity[i+3])
     return (ti,tx)

  def parsetetra(self,surface,connectivity):
    faces=[]
    for i in range(0,len(connectivity),4): 
      f1=self.getFace(surface,connectivity[i+0],connectivity[i+2],connectivity[i+1])
      f2=self.getFace(surface,connectivity[i+0],connectivity[i+1],connectivity[i+3])
      f3=self.getFace(surface,connectivity[i+1],connectivity[i+2],connectivity[i+3])
      f4=self.getFace(surface,connectivity[i+2],connectivity[i+0],connectivity[i+3])
      faces+=[f1,f2,f3,f4]
    return faces
          
  def parsehex(self,surface,connectivity):
    faces=[]
    for i in range(0,len(connectivity),8):                    
      f1=self.getFace(surface,connectivity[i+0],connectivity[i+3],connectivity[i+2],connectivity[i+1])
      f2=self.getFace(surface,connectivity[i+0],connectivity[i+1],connectivity[i+5],connectivity[i+4])
      f3=self.getFace(surface,connectivity[i+1],connectivity[i+2],connectivity[i+6],connectivity[i+5])
      f4=self.getFace(surface,connectivity[i+2],connectivity[i+3],connectivity[i+7],connectivity[i+6])
      f5=self.getFace(surface,connectivity[i+0],connectivity[i+4],connectivity[i+7],connectivity[i+3])
      f6=self.getFace(surface,connectivity[i+4],connectivity[i+5],connectivity[i+6],connectivity[i+7])
      faces+=[f1,f2,f3,f4,f5,f6]
    return faces

  def parsepyra(self,surface,connectivity):
    faces=[]
    for i in range(0,len(connectivity),5): 
      f1=self.getFace(surface,connectivity[i+0],connectivity[i+3],connectivity[i+2],connectivity[i+1])
      f2=self.getFace(surface,connectivity[i+0],connectivity[i+1],connectivity[i+4])
      f3=self.getFace(surface,connectivity[i+1],connectivity[i+2],connectivity[i+4])
      f4=self.getFace(surface,connectivity[i+2],connectivity[i+3],connectivity[i+4])
      f5=self.getFace(surface,connectivity[i+3],connectivity[i+0],connectivity[i+4])
      faces+=[f1,f2,f3,f4,f5]
    return faces  

  def def_volume(self,n):
    dic={2:(vtk.vtkVertex(),(1,1)),3:(vtk.vtkLine(),(2,2)),4:(vtk.vtkLine(),(2,3)),
         5:(vtk.vtkTriangle(),(3,3)),6:(vtk.vtkTriangle(),(3,6)),7:(vtk.vtkQuad(),(4,4)),
         8:(vtk.vtkQuad(),(4,8)),9:(vtk.vtkQuad(),(4,9)),10:(vtk.vtkTetra(),(4,4)),
         11:(vtk.vtkTetra(),(4,10)),12:(vtk.vtkPyramid(),(5,5)),13:(vtk.vtkPyramid(),(5,14)),
         14:(vtk.vtkPolyhedron(),(6,6)),15:(vtk.vtkPolyhedron(),(6,15)),16:(vtk.vtkPolyhedron(),(6,18)),
         17:(vtk.vtkHexahedron(),(8,8)),18:(vtk.vtkHexahedron(),(8,20)),19:(vtk.vtkHexahedron(),(8,27))}
    if dic.has_key(n):
      return dic[n]
    else:
      print "type non gere"          
      return False

#----------------------------------------------------------------------------

