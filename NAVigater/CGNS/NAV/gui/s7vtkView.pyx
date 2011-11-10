#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

cimport cython
cimport cpython
cimport numpy

import Tkinter
import math, os, sys
import vtk
import numpy
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
import numpy as NPY
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils    as CGU
import random
import s7globals
G___=s7globals.s7G

import s7windoz
import s7viewControl

# Lot of hard-coded stuff in there
# Now it works with both VTK and Cython,
# we should move to something more generic

#----------------------------------------------------------------------------
class CGNSparser:
  def __init__(self):
    pass
  def getActorList(self,T):
    ml=[]
    for m in self.parseZones(T):
      m.do_vtk()
      ml+=[m]
    return ml
  def parseZones(self,T):
    m=[]
    sl=[]
    ml=[]
    bl=[]
    mpath=[]
    spath=[]
    bpath=[]
    ncolors=0
    cl=G___.colors
    if (T[0]==None):
     T[0]=CGK.CGNSTree_s
     T[3]=CGK.CGNSTree_ts   
    for z in CGU.getAllNodesByTypeSet(T,[CGK.Zone_ts]):
      zT=CGU.nodeByPath(z,T)
      mpath.append(CGU.removeFirstPathItem(z))
      ncolors+=1
      g=CGU.getAllNodesByTypeSet(zT,[CGK.GridCoordinates_ts])[0]
      gT=CGU.nodeByPath(g,zT)
      cx=CGU.nodeByPath("%s/CoordinateX"%gT[0],gT)
      cy=CGU.nodeByPath("%s/CoordinateY"%gT[0],gT)
      cz=CGU.nodeByPath("%s/CoordinateZ"%gT[0],gT)
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
      m+=[[cx,cy,cz]]
      sl+=[[simin,simax,sjmin,sjmax,skmin,skmax]]
      zp=CGU.removeFirstPathItem(z)
      spath+=[[zp+'/[imin]',zp+'/[imax]',
               zp+'/[jmin]',zp+'/[jmax]',
               zp+'/[kmin]',zp+'/[kmax]']]
      for nzbc in CGU.getAllNodesByTypeSet(zT,[CGK.ZoneBC_ts]):
        zbcT=CGU.nodeByPath(nzbc,zT)
        for nbc in CGU.getAllNodesByTypeSet(zbcT,[CGK.BC_ts]):
          bpath+=[['%s/[%s]'%(zp,nbc.split('/')[1])]]
          bcT=CGU.nodeByPath(nbc,zbcT)
          for rbc in CGU.getAllNodesByTypeSet(bcT,[CGK.IndexRange_ts]):
            ptr=CGU.nodeByPath(rbc,bcT)[1].T.flat
            brg=scx[ptr[0]:ptr[3]+1,ptr[1]:ptr[4]+1,ptr[2]:ptr[5]+1]
            bl+=[[brg]]
      col=cl[cl.keys()[random.randrange(len(cl.keys()))]]
      if (len(bpath)>0):
        mc=Mesh(m[-1],mpath[-1],sl[-1],spath[-1],bl[-1],bpath[-1])
      else:
        mc=Mesh(m[-1],mpath[-1],sl[-1],spath[-1],[],[])
      mc.setColor(col)
      ml+=[mc]
    return ml

#----------------------------------------------------------------------------
class Mesh:

  def __init__(self,gcoordinates,gpath,surfaces,spaths,bcs,bcpaths):
    self.__surfs=surfaces
    self.__bnds=bcs
    self.dx=gcoordinates[0][1]
    self.dy=gcoordinates[1][1]
    self.dz=gcoordinates[2][1]
    self.imax=self.dx.shape[0]
    self.jmax=self.dx.shape[1]
    self.kmax=self.dx.shape[2]
    self.color=(130,130,130)
    self.actors=[]
    self.meshpath=gpath
    self.surfpaths=spaths
    self.bcpaths=bcpaths
    self.odict={}
    
  def setColor(self,color):
    self.color=color
    
#  @cython.boundscheck(False)
  def do_volume(self):
    cdef int p, i, j, k, idim, jdim, kdim
    cdef double x,y,z
    idim = self.imax
    jdim = self.jmax
    kdim = self.kmax
    pts=vtk.vtkPoints()
    pts.SetNumberOfPoints(self.imax*self.jmax*self.kmax)
    for i in range(idim):
     for j in range(jdim):
      for k in range(kdim):
       p=i+j*idim+k*idim*jdim
       x = (<double*>numpy.PyArray_GETPTR1(self.dx,p))[0]
       y = (<double*>numpy.PyArray_GETPTR1(self.dy,p))[0]
       z = (<double*>numpy.PyArray_GETPTR1(self.dz,p))[0]
       pts.InsertPoint(k+j*kdim+i*kdim*jdim,z,y,x)
    g=vtk.vtkStructuredGrid()
    self.odict[self.meshpath]=g
    g.SetPoints(pts)
    g.SetExtent(0,self.kmax-1,0,self.jmax-1,0,self.imax-1)
    d=vtk.vtkDataSetMapper()
    d.SetInput(g)
    a=vtk.vtkActor()
    self.actors.append(a)
    a.SetMapper(d)
    a.GetProperty().SetRepresentationToWireframe()
    a.GetProperty().SetColor(*self.color)

    self._bounds=a.GetBounds()
    return self

#  @cython.boundscheck(False)
  def do_surface(self,int n):
    cdef int i, j, imax, jmax, p1, p2, p3, p4
    imax=self.__surfs[n][0].shape[0]
    jmax=self.__surfs[n][0].shape[1]
    tx=self.__surfs[n][0].flat
    ty=self.__surfs[n][1].flat
    tz=self.__surfs[n][2].flat
    sg=vtk.vtkUnstructuredGrid()
    self.odict[self.surfpaths[n]]=sg
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
    self.actors.append(a)
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    a.GetProperty().SetColor(*self.color)
    return self
  def do_boundaries(self,n):
    print self.bcpaths[n],self.__bnds[n]
    return self
    cdef int i, j, imax, jmax, p1, p2, p3, p4
    imax=self.__bnds[n][0].shape[0]
    jmax=self.__bnds[n][0].shape[1]
    kmax=self.__bnds[n][0].shape[2]
    tx=self.__bnds[n][0].flat
    ty=self.__bnds[n][1].flat
    tz=self.__bnds[n][2].flat
    sg=vtk.vtkUnstructuredGrid()
    self.odict[self.surfpaths[n]]=sg
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
    self.actors.append(a)
    a.SetMapper(am)
    a.GetProperty().SetRepresentationToWireframe()
    a.GetProperty().SetColor(*self.color)
    return self
  def do_vtk(self):
    self.do_volume()
    for n in range(len(self.__surfs)):
      self.do_surface(n)
    for n in range(len(self.__bnds)):
      self.do_boundaries(n)

#----------------------------------------------------------------------------
class wVTKContext():
    def __init__(self,cm):
      (self.vx,self.vy,self.vz)=cm.GetViewUp()
      (self.cx,self.cy,self.cz)=cm.GetFocalPoint()
      (self.px,self.py,self.pz)=cm.GetPosition()
      (self.sx,self.sy,self.sz)=(0.,0.,0.)
      (self.ox,self.oy,self.oz)=(0.,0.,0.)
    def setViewUp(self,vx,vy,vz):
      (self.vx,self.vy,self.vz)=(vx,vy,vz)
    def setFocalPoint(self,cx,cy,cz):
      (self.cx,self.cy,self.cz)=(cx,cy,cz)
    def setPosition(self,px,py,pz):
      (self.px,self.py,self.pz)=(px,py,pz)
    def dump(self):
      return """
(self.vx,self.vy,self.vz)=%s
(self.cx,self.cy,self.cz)=%s
(self.px,self.py,self.pz)=%s
(self.sx,self.sy,self.sz)=%s
(self.ox,self.oy,self.oz)=%s"""%(\
        (self.vx,self.vy,self.vz),
        (self.cx,self.cy,self.cz),
        (self.px,self.py,self.pz),
        (self.sx,self.sy,self.sz),
        (self.ox,self.oy,self.oz))
     
class wVTKView(s7windoz.wWindoz):

  def SyncCameras(self,ren,event):
    cam = ren.GetActiveCamera()
    self.camAxes.SetViewUp(cam.GetViewUp())
    self.camAxes.OrthogonalizeViewUp()
    proj = cam.GetDirectionOfProjection()
    x, y, z = cam.GetDirectionOfProjection()

    # figure out the distance away from 0 0 0
    # if the renderer resets the camera to optimally inlcude all props
    # composing the orientation marker

    bnds = self.renAxes.ComputeVisiblePropBounds()
    x0, x1, y0, y1, z0, z1 = self.renAxes.ComputeVisiblePropBounds()
    self.renAxes.ResetCamera(x0, x1, y0, y1, z0, z1)
    pos = self.camAxes.GetPosition()
    px, py, pz = self.camAxes.GetPosition()
    d = numpy.sqrt(px*px + py*py + pz*pz)
    dproj = numpy.sqrt(x*x + y*y + z*z)

    # reset the camera back along the unit vector of the
    # direction of projection using our optimal distance
    self.camAxes.SetFocalPoint(0,0,0)
    self.camAxes.SetPosition(-d*x/dproj, -d*y/dproj, -d*z/dproj)
    self.renAxes.ResetCameraClippingRange()

  def findObjectPath(self,object):
    pth=''
    for m in self._meshes:
      for k in m.odict:
        if (m.odict[k]==object): pth+='%s'%k
    return pth
  
  def __init__(self,wparent,treefingerprint):

    self._xmin=self._ymin=self._zmin=self._xmax=self._ymax=self._zmax=0.0

    self._control=wparent
    s7windoz.wWindoz.__init__(self,wparent,'CGNS.NAV: VTK view')
    self._viewid=treefingerprint.addView(self,self._parent,'G')
    self._wtop.title('CGNS.NAV: [%s] VTK View [%.2d]'%\
                     (treefingerprint.filename,self._viewid))
    s7viewControl.updateWindowList(self._control._control,treefingerprint)

    self._vtktree=self.wCGNSTree(treefingerprint.tree)
    self._selected=[]
    self._currentactor=None
    self._viewtree=treefingerprint
    
  def leave(self):
    self._wtop.destroy()
    self._control._hasvtkWindow=None
  
  def onexit(self):
    self._control._control.delTreeView(self._viewid,
                                       self._viewtree.filedir,
                                       self._viewtree.filename)
    self._viewtree.hasWindow['G']=None
    self.leave()

  def annotatePick(self, object, event):
    self._selected=[]
    if self.picker.GetCellId() < 0:
        self.textActor.VisibilityOff()
    else:
        #selPt = self.picker.GetSelectionPoint()
        pickAct = self.picker.GetActors()
        pickAct.InitTraversal()
        a=pickAct.GetNextItem()
        t=''
        sl=[]
        while a:
          x=a.GetMapper().GetInput()
          s=self.findObjectPath(x)
          sl.append(s)
          t+=s+'\n'
          self._selected+=[(s,x)]
          a=pickAct.GetNextItem()
        self._control.clearAllMarks()
        self._control.updateMarkedFromList(sl)
        self.textMapper.SetInput(t)
        yd=self.rwin.GetSize()[1]-self.textMapper.GetHeight(self.wren)-10.
        self.textActor.SetPosition((10.,yd))
        self.textActor.VisibilityOn()

  def addPicker(self):
    self.textMapper = vtk.vtkTextMapper()
    tprop = self.textMapper.GetTextProperty()
    tprop.SetFontFamilyToArial()
    tprop.SetFontSize(10)
    tprop.BoldOff()
    tprop.ShadowOn()
    tprop.SetColor(0, 0, 1)
    self.textActor = vtk.vtkActor2D()
    self.textActor.VisibilityOff()
    self.textActor.SetMapper(self.textMapper)
    self.picker = vtk.vtkCellPicker()
    self.picker.SetTolerance(0.005)
    self.picker.AddObserver("EndPickEvent", self.annotatePick)
    
  def addAxis(self):
    self.camAxes = vtk.vtkCamera()
    self.camAxes.ParallelProjectionOn()

    self.renAxes = vtk.vtkRenderer()
    self.renAxes.InteractiveOff()
    self.renAxes.SetActiveCamera(self.camAxes)
    self.renAxes.SetViewport(0, 0, 0.2, 0.2)
    self.renAxes.SetBackground(1,1,1)

    xAxis = vtk.vtkArrowSource()
    xAxisMapper = vtk.vtkPolyDataMapper()
    xAxisMapper.SetInput(xAxis.GetOutput())
    xAxisActor = vtk.vtkActor()
    xAxisActor.SetMapper(xAxisMapper)
    xAxisActor.GetProperty().SetColor(1,0,0)

    yAxis = vtk.vtkArrowSource()
    yAxisMapper = vtk.vtkPolyDataMapper()
    yAxisMapper.SetInput(yAxis.GetOutput())
    yAxisActor = vtk.vtkActor()
    yAxisActor.SetMapper(yAxisMapper)
    yAxisActor.GetProperty().SetColor(1,1,0)
    yAxisActor.RotateZ(90)

    zAxis = vtk.vtkArrowSource()
    zAxisMapper = vtk.vtkPolyDataMapper()
    zAxisMapper.SetInput(zAxis.GetOutput())
    zAxisActor = vtk.vtkActor()
    zAxisActor.SetMapper(zAxisMapper)
    zAxisActor.GetProperty().SetColor(0,1,0)
    zAxisActor.RotateY(-90)

    xLabel = vtk.vtkCaptionActor2D()
    xLabel.SetCaption("X")
    xprop=vtk.vtkTextProperty()
    xprop.SetFontSize(3)
    xLabel.SetCaptionTextProperty(xprop)
    xLabel.SetAttachmentPoint(0.75,0.2,0)
    xLabel.LeaderOff()
    xLabel.BorderOff()
    xLabel.GetProperty().SetColor(0,0,0)
    xLabel.SetPosition(0,0)

    yLabel = vtk.vtkCaptionActor2D()
    yLabel.SetCaption("Y")
    yprop=vtk.vtkTextProperty()
    yprop.SetFontSize(3)
    yLabel.SetCaptionTextProperty(yprop)
    yLabel.SetAttachmentPoint(0.2,0.75,0)
    yLabel.LeaderOff()
    yLabel.BorderOff()
    yLabel.GetProperty().SetColor(0,0,0)
    yLabel.SetPosition(0,0)

    zLabel = vtk.vtkCaptionActor2D()
    zLabel.SetCaption("Z")
    zprop=vtk.vtkTextProperty()
    zprop.SetFontSize(3)
    zLabel.SetCaptionTextProperty(zprop)
    zLabel.SetAttachmentPoint(0,0.2,0.75)
    zLabel.LeaderOff()
    zLabel.BorderOff()
    zLabel.GetProperty().SetColor(0,0,0)
    zLabel.SetPosition(0,0)

    Axes3D = vtk.vtkPropAssembly()
    Axes3D.AddPart(xAxisActor)
    Axes3D.AddPart(yAxisActor)
    Axes3D.AddPart(zAxisActor)
    Axes3D.AddPart(xLabel)
    Axes3D.AddPart(yLabel)
    Axes3D.AddPart(zLabel)

    self.renAxes.AddActor(Axes3D)
    
    return self.renAxes

  def wCGNSTree(self,T):

      o=vtk.vtkObject()
      o.SetGlobalWarningDisplay(0)
      cpython.Py_DECREF(o)
      
      renWin = vtk.vtkRenderWindow()
      renWin.SetNumberOfLayers(2)
      wren = vtk.vtkRenderer()
      waxs = self.addAxis()
      wpck = self.addPicker()
      wren.AddActor2D(self.textActor)
      renWin.AddRenderer(wren)
      renWin.AddRenderer(waxs)
      wren.SetLayer(0)
      waxs.SetLayer(1)
      
      self._meshes=[]
      self._parser=CGNSparser()

      for m in self._parser.getActorList(T):
        a=m.actors
        self._meshes.append(m)
        for aa in a:
          wren.AddActor(aa)
      
        if (self._xmin>m._bounds[0]):self._xmin=m._bounds[0]
        if (self._ymin>m._bounds[2]):self._ymin=m._bounds[2]
        if (self._zmin>m._bounds[4]):self._zmin=m._bounds[4]
        if (self._xmax<m._bounds[1]):self._xmax=m._bounds[1]
        if (self._ymax>m._bounds[3]):self._ymax=m._bounds[3]
        if (self._zmax>m._bounds[5]):self._zmax=m._bounds[5]        

      wren.SetBackground(1,1,1)
      wren.ResetCamera()
      wren.GetActiveCamera().Elevation(0.0)
      wren.GetActiveCamera().Azimuth(90.0)
      wren.GetActiveCamera().Zoom(1.0)
      wren.GetActiveCamera().OrthogonalizeViewUp()
      
      (self.vx,self.vy,self.vz)=wren.GetActiveCamera().GetViewUp()
      (self.cx,self.cy,self.cz)=wren.GetActiveCamera().GetFocalPoint()
      (self.px,self.py,self.pz)=wren.GetActiveCamera().GetPosition()
      (self.sx,self.sy,self.sz)=(0.,0.,0.)
      (self.ox,self.oy,self.oz)=(0.,0.,0.)
      self._ctxt=wVTKContext(wren.GetActiveCamera())
      self._ctxt.setViewUp(self.vx,self.vy,self.vz)
      self._ctxt.setFocalPoint(self.cx,self.cy,self.cz)
      self._ctxt.setPosition(self.px,self.py,self.pz)

      iwin = vtkTkRenderWindowInteractor(self._wtop,width=500,height=500,
                                         rw=renWin)
      iwin.pack(expand="t", fill="both")

      iren = renWin.GetInteractor()
      istyle = vtk.vtkInteractorStyleTrackballCamera()
      iren.SetInteractorStyle(istyle)
      iren.AddObserver("KeyPressEvent", self.CharCallback)
      iren.SetPicker(self.picker)
      wren.AddObserver("StartEvent", self.SyncCameras)

      iren.Initialize()
      renWin.Render()
      iren.Start()

      self.iren=iren
      self.wren=wren
      self.waxs=waxs
      self.rwin=renWin
      self.isty=istyle
      self.iwin=iwin

      self._bindings={ 'space' :self.b_refresh,
                       'c'     :self.b_shufflecolors,
                       'Tab'   :self.b_nexttarget,
                       'x'     :self.b_xaxis,
                       'y'     :self.b_yaxis,
                       'z'     :self.b_zaxis,
                       'X'     :self.b_xaxis_flip,
                       's'     :self.b_surf,
                       'w'     :self.b_wire }

      self._p_wire=True
      return self.rwin

  def b_shufflecolors(self,pos):
    actors = self.wren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    cl=G___.colors
    while actor:
      col=cl[cl.keys()[random.randrange(len(cl.keys()))]]
      actor.GetProperty().SetColor(col)
      actor = actors.GetNextItem()
    self.rwin.Render() 
      
  def b_nexttarget(self,pos):
    if (len(self._selected)>1):
      self._selected=self._selected[1:]+[self._selected[0]]
    if (self._currentactor==None):
      self._currentactor=self._selected[0]      
    self.rwin.Render()
      
  def b_refresh(self,pos):
    self.rwin.Render()
      
  def b_xaxis(self,pos):
    self.setAxis(pos,1)
    
  def b_yaxis(self,pos):
    self.setAxis(pos,2)
    
  def b_zaxis(self,pos):
    self.setAxis(pos,3)
    
  def b_xaxis_flip(self,pos):
    self.setAxis(pos,-1)

  def b_yaxis_flip(self,pos):
    self.setAxis(pos,-2)

  def b_zaxis_flip(self,pos):
    self.setAxis(pos,-3)

  def setAxis(self,pos,iaxis):
    rat=50
    cx = (0.5*(self._xmax-self._xmin))
    cy = (0.5*(self._ymax-self._ymin))
    cz = (0.5*(self._zmax-self._zmin))
    if iaxis == 1:
      (vx,vy,vz)=(0.,0.,1.)
      (px,py,pz)=(rat*cx,cy,cz)
    elif iaxis == 2:
      (vx,vy,vz)=(0.5,0.,0.)
      (px,py,pz)=(cx,rat*cy,cz)
    elif iaxis == 3:
      (vx,vy,vz)=(0.,1.,0.)
      (px,py,pz)=(cx,cy,rat*cz)
    elif iaxis == -1:
      (vx,vy,vz)=(0.,0.,-1.)
      (px,py,pz)=(rat*cx,cy,cz)
    elif iaxis == -2:
      (vx,vy,vz)=(-0.5,0.,0.)
      (px,py,pz)=(cx,rat*cy,cz)
    elif iaxis == -3:
      (vx,vy,vz)=(0.,-1.,0.)
      (px,py,pz)=(cx,cy,rat*cz)

    camera = self.wren.GetActiveCamera()
    camera.SetViewUp(vx, vy, vz)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetPosition(px, py, pz)
    camera.OrthogonalizeViewUp()
    self.wren.ResetCameraClippingRange()
    self.wren.Render()
    self.waxs.Render()
    self.wren.ResetCamera()
    self.iren.Render()

    self._ctxt=wVTKContext(camera)
    self._ctxt.setViewUp(vx,vy,vz)
    self._ctxt.setFocalPoint(cx,cy,cz)
    self._ctxt.setPosition(px,py,pz)

  def b_surf(self,pos):
    if (not self._p_wire):
      self.b_wire(pos)
      return
    self._p_wire=False
    actors = self.wren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToSurface()
        actor = actors.GetNextItem()
    self.rwin.Render()
      
  def b_wire(self,pos):
    if (self._p_wire):
      self.b_surf(pos)
      return
    self._p_wire=True
    actors = self.wren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToWireframe()
        actor = actors.GetNextItem()
    self.rwin.Render() 
      
  def MotionCallback(self,obj,event):
      pos = self.iren.GetEventPosition()
      print 'MOTION AT ',pos
      return

  def CharCallback(self,obj,event):
      iren = self.rwin.GetInteractor()
      keysym  = iren.GetKeySym()
      pos = self.iren.GetEventPosition()
      if (self._bindings.has_key(keysym)): self._bindings[keysym](pos)
      return 

# -----------------------------------------------------------------------------
# --- last line


