#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.Q7VTKWindow import Ui_Q7VTKWindow
from CGNS.NAV.wfingerprint import Q7Window
from CGNS.NAV.mparser import Mesh
from CGNS.NAV.moption import Q7OptionContext as OCTXT
import numpy as NPY

import random
import vtk

# ----------------------------------------------------------------------------
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

# -----------------------------------------------------------------
class Q7VTK(Q7Window,Ui_Q7VTKWindow):
  def __init__(self,control,node,fgprint,tmodel,zlist):
      if (not zlist): pth='/'
      else: pth='<partial>'
      Q7Window.__init__(self,Q7Window.VIEW_VTK,control,pth,fgprint)
      self._xmin=self._ymin=self._zmin=self._xmax=self._ymax=self._zmax=0.0
      self._epix=QIcon(QPixmap(":/images/icons/empty.gif"))
      self._spix=QIcon(QPixmap(":/images/icons/selected.gif"))
      self._npix=QIcon(QPixmap(":/images/icons/unselected.gif"))
      self._hpix=QIcon(QPixmap(":/images/icons/hidden.gif"))
      self._T=self._fgprint.tree
      self._selected=[]
      self._hidden=[]
      self._cacheActor={}
      self._tmodel=tmodel
      self.color=(0,0,0)
      self.OutlineActor=None
      self._currentactor=None
      self.PickedRenderer=None
      self.PropPicked=0
      self.controlKey=0
      self.highlightProp=None
      self._camera={}
      self._blackonwhite=False
      self._vtktree=self.wCGNSTree(self._fgprint.tree,zlist)
      self.display.Initialize()
      self.display.Start()
      self.display.show()
      self.bX.clicked.connect(self.b_xaxis)
      self.bY.clicked.connect(self.b_yaxis)
      self.bZ.clicked.connect(self.b_zaxis)
      self.bReverse.clicked.connect(self.reverseSelection)
      self.bSaveVTK.clicked.connect(self.b_saveVTK)
      self.bSuffleColors.clicked.connect(self.b_shufflecolors)
      self.bBlackColor.clicked.connect(self.b_blackandwhite)
      self.bAddView.clicked.connect(self.b_saveview)
      self.bRemoveView.clicked.connect(self.b_delview)
      self.bNext.clicked.connect(self.b_next)
      self.bPrevious.clicked.connect(self.b_prev)
      self.bReset.clicked.connect(self.b_reset)
      self.bUpdate.clicked.connect(self.b_update)
      QObject.connect(self.cViews,
                      SIGNAL("currentIndexChanged(int)"),
                      self.b_loadview)
      QObject.connect(self.cViews.lineEdit(),
                      SIGNAL("editingFinished()"),
                      self.b_saveview)
      QObject.connect(self.cCurrentPath,
                      SIGNAL("currentIndexChanged(int)"),
                      self.changeCurrentPath)
   
  def SyncCameras(self,ren,event):
    cam = ren.GetActiveCamera()
    self.camAxes.SetViewUp(cam.GetViewUp())
    self.camAxes.OrthogonalizeViewUp()
    x, y, z = cam.GetDirectionOfProjection()
    bnds = self.renAxes.ComputeVisiblePropBounds()
    x0, x1, y0, y1, z0, z1 = self.renAxes.ComputeVisiblePropBounds()
    self.renAxes.ResetCamera(x0, x1, y0, y1, z0, z1) 
    pos = self.camAxes.GetPosition()
    px, py, pz = self.camAxes.GetPosition()
    d = NPY.sqrt(px*px + py*py + pz*pz)
    dproj = NPY.sqrt(x*x + y*y + z*z)
    self.camAxes.SetFocalPoint(0,0,0)
    self.camAxes.SetPosition(-d*x/dproj, -d*y/dproj, -d*z/dproj)
    self.renAxes.ResetCameraClippingRange()

  def SyncCameras2(self,ren,event):
    cam=ren.GetActiveCamera()
    self.camforRen.SetViewUp(cam.GetViewUp())
    self.camforRen.SetFocalPoint(cam.GetFocalPoint())
    self.camforRen.SetPosition(cam.GetPosition())
    self.camforRen.SetClippingRange(cam.GetClippingRange())
    self.camforRen.SetParallelScale(cam.GetParallelScale())
    self.camforRen.SetViewAngle(cam.GetViewAngle())
    self.camforRen.SetParallelProjection(cam.GetParallelProjection())

  def findObjectPath(self,selected):
    return self._parser.getPathFromObject(selected)
    
  def findPathObject(self,path):
    if (self._cacheActor.has_key(path)): return self._cacheActor[path]
    alist=self._vtkren.GetActors()
    alist.InitTraversal()
    a=alist.GetNextItem()
    while a:
        if (path==self.findObjectPath(a.GetMapper().GetInput())):
            self._cacheActor[path]=a
            return a
        a=alist.GetNextItem()
    return None
    
  def leave(self):
    self._wtop.destroy()
    self._control._hasvtkWindow=None
  
  def onexit(self):  
    self._control._control.delTreeView(self._viewid,
                                       self._fgprint.filedir,
                                       self._fgprint.filename)
    self.leave() 

  def foregroundRenderer(self):
    self.camforRen = vtk.vtkCamera()
    self.camforRen.ParallelProjectionOn()
    self.renforRen = vtk.vtkRenderer()
    self.renforRen.InteractiveOff()
    self.renforRen.SetActiveCamera(self.camforRen)
    self.renforRen.SetViewport(0, 0, 1, 1)
    self.renforRen.SetBackground(1,1,1)      

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

    self.xLabel=xLabel
    self.yLabel=yLabel
    self.zLabel=zLabel
    
    return self.renAxes

  def colourLabel(self):
      self.xLabel.GetProperty().SetColor(self.color)
      self.yLabel.GetProperty().SetColor(self.color)
      self.zLabel.GetProperty().SetColor(self.color)
    
  def addPicker(self):
      self.textMapper = vtk.vtkTextMapper()
      tprop = self.textMapper.GetTextProperty()  
      tprop.SetFontFamilyToArial()
      tprop.SetFontSize(10)
      tprop.BoldOff()
      tprop.ShadowOn()
      tprop.SetColor(0, 0, 0)
      self.textActor = vtk.vtkActor2D()
      self.textActor.VisibilityOff()
      self.textActor.SetMapper(self.textMapper)
      self.picker = vtk.vtkCellPicker()
      self.picker.SetTolerance(0.001)
      self.picker.AddObserver("EndPickEvent",self.annotatePick)

  def annotatePick(self,object,event):
      if (self.controlKey!=1):
        self._selected=[]
        self.PropPicked=0
      if (self.picker.GetCellId()<0):
        self.textActor.VisibilityOff()
        self.highlightProp=0
      else:
        selPt = self.picker.GetSelectionPoint()
        pickAct = self.picker.GetActors()
        pickAct.InitTraversal()
        a=pickAct.GetNextItem()        
        t=''
        sl=[]
        sz=self._iren.GetRenderWindow().GetSize()[1]
        s=None
        while a:
          x=a.GetMapper().GetInput()
          s=self.findObjectPath(x)
          t+=s+'\n'
          self._selected+=[[s,a]]
          self.textMapper.SetInput(t)
          y=sz-self.textMapper.GetHeight(self._vtkren)-10.
          self.textActor.SetPosition((10.,y))
          self.textActor.VisibilityOn()
          self._vtkren.AddActor(self.textActor)
          a=pickAct.GetNextItem()
        self.PropPicked=0
        self.controlKey=0
        self.highlightProp=1
        

  def wCGNSTree(self,T,zlist):

      o=vtk.vtkObject()
      o.SetGlobalWarningDisplay(0)
      del o
      
      self._vtk=self.display
      self._vtk.setParent(self)
      self._vtkren = vtk.vtkRenderer()
      
      self._waxs=self.addAxis()
      self._vtkwin=self._vtk.GetRenderWindow()
      self._vtkwin.SetNumberOfLayers(3)
      self._vtkren.SetLayer(0)
      self._waxs.SetLayer(2)

      self.foregroundRenderer()
      self.renforRen.SetLayer(1)

      self._vtk.GetRenderWindow().AddRenderer(self._vtkren)
      self._vtk.GetRenderWindow().AddRenderer(self._waxs)
      self._vtk.GetRenderWindow().AddRenderer(self.renforRen)
      
      self._parser=Mesh(T,zlist)
      self._selected=[]
      alist=self._parser.createActors()
 
      if (alist!=[]):
          for a in alist:
              self._vtkren.AddActor(a[0])

              if (a[1] is not None):
                if (self._xmin>a[1][0]):self._xmin=a[1][0]
                if (self._ymin>a[1][2]):self._ymin=a[1][2]
                if (self._zmin>a[1][4]):self._zmin=a[1][4]
                if (self._xmax<a[1][1]):self._xmax=a[1][1]
                if (self._ymax<a[1][3]):self._ymax=a[1][3]
                if (self._zmax<a[1][5]):self._zmax=a[1][5]
                
      self._vtkren.SetBackground(1,1,1)
      self._vtkren.ResetCamera()
      self._vtkren.GetActiveCamera().Elevation(0.0)
      self._vtkren.GetActiveCamera().Azimuth(90.0)
      self._vtkren.GetActiveCamera().Zoom(1.0)
##       self._vtkren.GetActiveCamera().OrthogonalizeViewUp()
##       self._vtkren.GetActiveCamera().SetViewAngle(0)
      self._vtkren.ResetCamera()
                        
      (self.vx,self.vy,self.vz)=self._vtkren.GetActiveCamera().GetViewUp()
      (self.cx,self.cy,self.cz)=self._vtkren.GetActiveCamera().GetFocalPoint()
      (self.px,self.py,self.pz)=self._vtkren.GetActiveCamera().GetPosition()
      (self.sx,self.sy,self.sz)=(0.,0.,0.)
      (self.ox,self.oy,self.oz)=(0.,0.,0.)

      self._ctxt=wVTKContext(self._vtkren.GetActiveCamera())
      self._ctxt.setViewUp(self.vx,self.vy,self.vz)
      self._ctxt.setFocalPoint(self.cx,self.cy,self.cz)
      self._ctxt.setPosition(self.px,self.py,self.pz)
                
      self._istyle=Q7InteractorStyle(self)
      self._vtk.SetInteractorStyle(self._istyle)
      self._iren=self._istyle.GetInteractor()
      self.fillCurrentPath()
      
      self._vtkren.AddObserver("StartEvent", self.SyncCameras)
      self._vtkren.AddObserver("StartEvent", self.SyncCameras2)
      
      self._p_wire=True
      self.setColors(True)

      self._bindings={ 's'     :self.b_surf,
                       'w'     :self.b_wire,
                       'r'     :self.resetCam,
                       'd'     :self.hideActor}
      
      return self._vtk.GetRenderWindow()

  def b_shufflecolors(self,pos=None):
      self.setColors(True)
      
  def b_blackandwhite(self,pos=None):
      self.setColors()
      
  def getRandomColor(self):
      clst=OCTXT._ColorList
      cl=clst[clst.keys()[random.randrange(len(clst.keys()))]]
      return cl
      
  def setColors(self,randcolors=False):
      if (not randcolors):
          if (self._blackonwhite):
              self._vtkren.SetBackground(1,1,1)
              self.color=(0,0,0)
              self.colourLabel()
              cl=(0,0,0)
          else:
              self._vtkren.SetBackground(0,0,0)
              self.color=(1,1,1)
              self.colourLabel()
              cl=(1,1,1)
          self._blackonwhite=not self._blackonwhite
      actors = self._vtkren.GetActors()
      actors.InitTraversal()
      actor = actors.GetNextItem()
      while actor:
          if (randcolors): cl=self.getRandomColor()
          actor.GetProperty().SetColor(cl)
          actor = actors.GetNextItem()
      if (not self.OutlineActor is None):
          self.OutlineActor.GetProperty().SetColor(0,1,1)
      if (self.highlightProp==1):
          color=self._currentactor[1].GetProperty().GetColor()
          self._currentactor[1].GetProperty().SetColor(1,0,0)
          self._currentactor[2]=color
      else:
          self._currentactor=None
      self._iren.Render()

  def b_update(self):
      self.busyCursor()
      self._selected=[]
      self.textActor.VisibilityOff()
      tlist=self._tmodel.getSelectedShortCut()
      slist=self._parser.getPathList()
      for i in tlist:
          if (i in slist):
              self._selected+=[[i,self.findPathObject(i)]]
      self.fillCurrentPath()
      self.readyCursor()
      
  def b_reset(self):
      self._selected=[]
      for i in self._hidden:
          i[1].VisibilityOn()
      self._hidden=[]
      self._selected=[]
      self.textActor.VisibilityOff()
      self.changeCurrentActor([None,None])
      self.fillCurrentPath()
      
  def b_next(self):
      if (len(self._selected)>0):        
          self._selected=self._selected[1:]+[self._selected[0]]
          self.PropPicked=1
          self.changeCurrentActor(self._selected[0],False)
          return self._selected[0]
      
  def b_prev(self):
      if (len(self._selected)>0):        
          self._selected=[self._selected[-1]]+self._selected[0:-1]
          self.PropPicked=1
          self.changeCurrentActor(self._selected[0],False)
      
  def b_loadview(self,name=None):
      vname=self.cViews.currentText()
      if (self._camera.has_key(vname)):
          (vu,cr,p,fp,va,ps,pj)=self._camera[vname]
          camera=self._vtkren.GetActiveCamera()
          camera.SetViewUp(vu)
          camera.SetClippingRange(cr)
          camera.SetPosition(p)
          camera.SetFocalPoint(fp)
          camera.SetViewAngle(va)
          camera.SetParallelScale(ps)
          camera.SetParallelProjection(pj)
          self._iren.Render()

  def updateViewList(self):
    k=self._camera.keys()
    k.sort()
    self.cViews.clear()
    self.cViews.addItem("")
    for i in k: self.cViews.addItem(i)
    self.cViews.setCurrentIndex(0)
        
  def b_delview(self,name=None):
    name=str(self.cViews.currentText())
    if ((name=="") or (not self._camera.has_key(name))): return
    del self._camera[name]
    self.updateViewList()
    
  def b_saveview(self,name=None):
    camera=self._vtkren.GetActiveCamera()
    name=str(self.cViews.currentText())
    if ((name=="") or (self._camera.has_key(name))): return
    self._camera[name]=(camera.GetViewUp(),
                        camera.GetClippingRange(),
                        camera.GetPosition(),
                        camera.GetFocalPoint(),
                        camera.GetViewAngle(),
                        camera.GetParallelScale(),
                        camera.GetParallelProjection()
                        )
    self.updateViewList()

  def b_refresh(self,pos):
      self._vtk.GetRenderWindow().Render()
      
  def b_saveVTK(self,*args):
      w=vtk.vtkGenericDataObjectWriter()
      w.SetFileName('/tmp/Foo.vtk')
      actors = self._vtkren.GetActors()
      actors.InitTraversal()
      actor = actors.GetNextItem()
      while actor:
          w.SetInput(actor.GetMapper().GetInput())
          actor = actors.GetNextItem()
      w.Write()

  def b_xaxis(self,pos=None):
      if (self.cMirror.isChecked()): self.setAxis(pos,-1)
      else: self.setAxis(pos,1)
    
  def b_yaxis(self,pos=None):
      if (self.cMirror.isChecked()): self.setAxis(pos,-2)
      else: self.setAxis(pos,2)
   
  def b_zaxis(self,pos=None):
      if (self.cMirror.isChecked()): self.setAxis(pos,-3)
      else: self.setAxis(pos,3)
    
  def changeCurrentPath(self, *args):
      path=self.cCurrentPath.currentText()
      if (path==''):
          return
      if (self.PropPicked==1):
          self.PropPicked=0
          return
      actor=self.findPathObject(path)
      self.changeCurrentActor([path,actor])
      
  def setCurrentPath(self,path):
      ix=self.cCurrentPath.findText(path)
      if (ix!=-1):
          self.cCurrentPath.setCurrentIndex(ix)

  def fillCurrentPath(self):
      self.cCurrentPath.clear()
      sel=[n[0] for n in self._selected]
      hid=[n[0] for n in self._hidden]
      self.cCurrentPath.addItem(self._epix,'')
      for i in self._parser.getPathList():
          pix=self._npix
          if (i in sel): pix=self._spix
          if (i in hid): pix=self._hpix
          self.cCurrentPath.addItem(pix,i)
      self._iren.Render()

  def reverseSelection(self):
      selected=[]
      selection=[]
      hidden=[]
      for i in self._selected:
          selected.append(i[0])
      for i in self._hidden:
          hidden.append(i[0])
      for i in self._parser.getPathList():
          if ((not i in selected) and (not i in hidden)):
              selection.append([i,self.findPathObject(i)])
      self._selected=selection
      actor=self.b_next()
      path=actor[0]
      self.fillCurrentPath()
      if (path is not None):
          self.setCurrentPath(path)

  def setAxis(self,pos,iaxis):
    camera=self._vtkren.GetActiveCamera()
    fp=camera.GetFocalPoint()
    pos=camera.GetPosition()
    distance=NPY.sqrt((fp[0]-pos[0])*(fp[0]-pos[0])
                      +(fp[1]-pos[1])*(fp[1]-pos[1])
                      +(fp[2]-pos[2])*(fp[2]-pos[2]))
    if iaxis == 1:
      (vx,vy,vz)=(0.,0.,1.)
      (px,py,pz)=(fp[0]+distance,fp[1],fp[2])
    elif iaxis == 2:
      (vx,vy,vz)=(0.,0.,1.)
      (px,py,pz)=(fp[0],fp[1]+distance,fp[2])
    elif iaxis == 3:
      (vx,vy,vz)=(0.,1.,0.)
      (px,py,pz)=(fp[0],fp[1],fp[2]+distance)
    elif iaxis == -1:
      (vx,vy,vz)=(0.,0.,1.)
      (px,py,pz)=(fp[0]-distance,fp[1],fp[2])
    elif iaxis == -2:
      (vx,vy,vz)=(0.,0.,1.)
      (px,py,pz)=(fp[0],fp[1]-distance,fp[2])
    elif iaxis == -3:
      (vx,vy,vz)=(0.,1.,0.)
      (px,py,pz)=(fp[0],fp[1],fp[2]-distance)
    camera.SetViewUp(vx,vy,vz)
    camera.SetPosition(px,py,pz)
    camera.SetViewAngle(0)
    self._vtkren.ResetCameraClippingRange()
    self._vtkren.Render()
    self._waxs.Render()
    self._vtkren.ResetCamera()
    self._iren.Render()
    self._ctxt=wVTKContext(camera)
    self._ctxt.setViewUp(vx,vy,vz)
    self._ctxt.setPosition(px,py,pz)
          
  def b_surf(self,pos):
      if (not self._p_wire):
          self.b_wire(pos)
          return
      self._p_wire=False
      actors = self._vtkren.GetActors()
      actors.InitTraversal()
      actor = actors.GetNextItem()
      while actor:
          actor.GetProperty().SetRepresentationToSurface()
          actor = actors.GetNextItem()
      self._vtk.GetRenderWindow().Render()
      
  def b_wire(self,pos):  
      if (self._p_wire):
          self.b_surf(pos)
          return
      self._p_wire=True
      actors = self._vtkren.GetActors() 
      actors.InitTraversal()
      actor = actors.GetNextItem()
      while actor:
          actor.GetProperty().SetRepresentationToWireframe()
          actor = actors.GetNextItem()
      self._vtk.GetRenderWindow().Render()

  def hideActor(self,pos):
      if (not self._currentactor is None):
          if (self._currentactor[0:2] in self._selected):
              self._currentactor[1].VisibilityOff()
              self._selected.remove(self._currentactor[0:2])
              self._hidden.append(self._currentactor[0:2])
              actor=self.b_next()
              self.fillCurrentPath()
              if (actor is not None):
                  s=self.findObjectPath(actor[1].GetMapper().GetInput())
                  if (s is not None):
                      self.setCurrentPath(s)
              else:
                  self.changeCurrentActor([None,None])

  def resetCam(self,pos):
      self._vtkren.ResetCamera()
      self._iren.Render()
        
  def close(self):
      self._vtk.GetRenderWindow().Finalize()
      QWidget.close(self)

  def changeCurrentActor(self,atp,combo=True):
      path =atp[0]
      actor=atp[1]
      if (not combo):self.setCurrentPath(path)
      if (not self._currentactor is None):
          act=self._currentactor[1]
          col=self._currentactor[2]
          self.renforRen.RemoveActor(act)
          act.GetProperty().SetColor(col)
      if (actor is None):
          self.cCurrentPath.setCurrentIndex(0)
          if (not self.PickedRenderer is None and not self.OutlineActor is None):
              self.PickedRenderer.RemoveActor(self.OutlineActor)
              self.PickedRenderer=None
              self.CurrentRenderer.Render()
              self.CurrentRenderer.GetRenderWindow().Render()
          return
      color=actor.GetProperty().GetColor()
      actor.GetProperty().SetColor(1,0,0)
      self.renforRen.AddActor(actor)
      self.CurrentRenderer=self.renforRen
      self.Outline=vtk.vtkOutlineSource()
      self.OutlineMapper=vtk.vtkPolyDataMapper()
      self.OutlineMapper.SetInput(self.Outline.GetOutput())
      if (not self.PickedRenderer is None and not self.OutlineActor is None):
          self.PickedRenderer.RemoveActor(self.OutlineActor)
          self.PickedRenderer=None
      if (not actor is None):
          self.OutlineActor=vtk.vtkActor()
          self.OutlineActor.PickableOff()
          self.OutlineActor.DragableOff()
          self.OutlineActor.SetMapper(self.OutlineMapper)
          self.OutlineActor.GetProperty().SetColor(0,1,1)
          self.OutlineActor.GetProperty().SetAmbient(1.0)
          self.OutlineActor.GetProperty().SetDiffuse(0.0)
          self.OutlineActor.GetProperty().SetLineWidth(1.2)
          if (self.CurrentRenderer!=self.PickedRenderer):
              if (not self.PickedRenderer is None and not self.OutlineActor is None):
                  self.PickedRenderer.RemoveActor(self.OutlineActor)
              self.CurrentRenderer.AddActor(self.OutlineActor)
              self.PickedRenderer=self.CurrentRenderer
          self.Outline.SetBounds(actor.GetBounds())
      self._iren.Render()
      self._currentactor=[path,actor,color]

class Q7InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self,parent):
      self._parent=parent
##       self.AddObserver("KeyPressEvent", self.CharCallback)
##       self.AddObserver("KeyPressEvent",self.setPicker)
##       self.AddObserver("KeyReleaseEvent",self.setPicker)
##       self.AddObserver("CharEvent",self.setPicker)
      self.AddObserver("CharEvent",self.keycode)      
      self.PickedRenderer=None
      self.OutlineActor=None
      self._parent.addPicker()     
      self.rwi=None  

    def keycode(self,*args):
        self.rwi=self.GetInteractor()      
        keycode=self.rwi.GetKeyCode()
        control=self.rwi.GetControlKey()
        vtkkeys=['f','F','s','S','w','W','r','R']
        if keycode in vtkkeys:
            self.OnChar()
        keys={'p':self.setPick,'d':self.CharCallback}
        if (keys.has_key(keycode)):
            keys[keycode]()
        if (control==1): self._parent.controlKey=1
               
    def setPick(self):
        act=None
        path=None
        eventPos=self.rwi.GetEventPosition()
        self.rwi.SetPicker(self._parent.picker)
        if (not self._parent.picker is None):
          self._parent.picker.Pick(eventPos[0],eventPos[1], 
                       0.0,self._parent._vtkren)
          p=self._parent.picker.GetPath()
          if (p is None):
              self._parent.changeCurrentActor([None,None])
              self._parent.PropPicked=0
          else:
              actor=p.GetFirstNode().GetViewProp()
              path=self._parent.findObjectPath(actor.GetMapper().GetInput())
              self._parent.changeCurrentActor([path,actor])
              self._parent.PropPicked=1
          self._parent.fillCurrentPath()
          if (path is not None):
            self._parent.setCurrentPath(path)

    def CharCallback(self,*args):   
      keycode=self.rwi.GetKeyCode()
      control=self.rwi.GetControlKey()
      pos=self.rwi.GetEventPosition()
      if (self._parent._bindings.has_key(keycode)): self._parent._bindings[keycode](pos)
      if (control==1): self._parent.controlKey=1     
      return
    
##     def setPicker(self,*args):
##       self.rwi=self.GetInteractor()
##       keycode=self.rwi.GetKeyCode()
##       if (keycode=='p'):
##         self.rwi.SetKeyCode('')
##         act=None
##         path=None
##         eventPos=self.rwi.GetEventPosition()
##         self.FindPokedRenderer(eventPos[0],eventPos[1])
##         self.rwi.StartPickCallback()
##         self.rwi.SetPicker(self._parent.picker)
##         if (not self._parent.picker is None):
##           self._parent.picker.Pick(eventPos[0],eventPos[1], 
##                        0.0,self.GetCurrentRenderer())
##           p=self._parent.picker.GetPath()
##           if (p is None):
##               self._parent.changeCurrentActor([None,None])
##               self._parent.PropPicked=0
##           else:
##               actor=p.GetFirstNode().GetViewProp()
##               path=self._parent.findObjectPath(actor.GetMapper().GetInput())
##               self._parent.changeCurrentActor([path,actor])
##               self._parent.PropPicked=1
##           self._parent.fillCurrentPath()
##           if (path is not None):
##             self._parent.setCurrentPath(path)
##         self.rwi.EndPickCallback()
        






