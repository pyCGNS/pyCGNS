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
from CGNS.NAV.wfile import Q7File
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
      self.lut=None
      self.grid_dims=None
      self.mincolor=None
      self.maxcolor=None
      self.scalarbar=None
      self._selected=[]
      self._picked=None
      self._hidden=[]
      self.actorpt=None
      self.wact=None
      self._cacheActor={}
      self._tmodel=tmodel
      self.color=(0,0,0)
      self.OutlineActor=None
      self._currentactor=None
      self.PickedRenderer=None
      self.PropPicked=0
      self.controlKey=0
      self.highlightProp=None
      self.maxColorMap=None
      self._camera={}
      self._blackonwhite=False
      self._vtktree=self.wCGNSTree(self._fgprint.tree,zlist)
      self.resetSpinBox()
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
      self.bScreenShot.clicked.connect(self.screenshot)
      self.ColorMapMin=QColorDialog(self)
      self.ColorMapMax=QColorDialog(self)      
      QObject.connect(self.ColorMapMin, SIGNAL("colorSelected(QColor)"), self.getColorMapMin)
      QObject.connect(self.ColorMapMax, SIGNAL("colorSelected(QColor)"), self.getColorMapMax)            
      self.bColorMapMin.clicked.connect(self.displayColorMapMin)
      self.bColorMapMax.clicked.connect(self.displayColorMapMax)
      self.setColorSpace()
      QObject.connect(self.cViews,
                      SIGNAL("currentIndexChanged(int)"),
                      self.b_loadview)
      QObject.connect(self.cViews.lineEdit(),
                      SIGNAL("editingFinished()"),
                      self.b_saveview)
      QObject.connect(self.cCurrentPath,
                      SIGNAL("currentIndexChanged(int)"),
                      self.changeCurrentPath)
      QObject.connect(self.sIndex1, SIGNAL("valueChanged(int)"),
                      self.highlightPoint)
      QObject.connect(self.sIndex2, SIGNAL("valueChanged(int)"),
                      self.highlightPoint)
      QObject.connect(self.sIndex3, SIGNAL("valueChanged(int)"),
                      self.highlightPoint)
##       self.setCursor(Qt.WaitCursor)
##       self.unsetCursor()

  def displayColorMapMin(self,*args):
      self.ColorMapMin.show()

  def displayColorMapMax(self,*args):
      self.ColorMapMax.show()

  def getColorMapMin(self,*args):
      self.maxcolor=self.ColorMapMin.currentColor().getRgbF()

  def getColorMapMax(self,*args):
      self.mincolor=self.ColorMapMax.currentColor().getRgbF()

  def setVariable(self,variable=None):
      self.cVariables.clear()
      self.cVariables.addItem("")
      if (variable!=[]):
          for i in variable:
              self.cVariables.addItem(i)
      QObject.connect(self.cVariables,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getVariable)
      
  def getVariable(self,*args):
      if ((self.mincolor is None) or (self.maxcolor is None)): return
      variable=self.cVariables.currentText()
      index=self.cVariables.currentIndex()
      self.scalarbar.SetTitle(variable)
      if (index==0):
          if (not self.scalarbar is None):
              self.scalarbar.SetVisibility(0)
              self.setScalarVisibilityOff()
          return
      else :
          a=self._vtkren.GetActors()
          a.InitTraversal()
          s=a.GetNextItem()
          minrange=[]
          maxrange=[]
          while s:
              grid=s.GetMapper().GetInput()
              if (vtk.vtkStructuredGrid.SafeDownCast(grid)):
                  grid.GetCellData().SetActiveScalars(variable)
                  minrange+=[grid.GetScalarRange()[0]]
                  maxrange+=[grid.GetScalarRange()[1]]
                  s.GetMapper().SetScalarModeToUseCellData()
                  s.GetMapper().ScalarVisibilityOn()
                  if (not self.lut is None):
                      s.GetMapper().SetLookupTable(self.lut)
                      self.lut.RemoveAllPoints()
                      self.lut.AddRGBPoint(min(minrange),self.mincolor[0],self.mincolor[1],self.mincolor[2])
                      self.lut.AddRGBPoint(max(maxrange),self.maxcolor[0],self.maxcolor[1],self.maxcolor[2])
              s=a.GetNextItem()
      self.scalarbar.SetVisibility(1)
      self._iren.Render()

  def setScalarVisibilityOff(self):
      a=self._vtkren.GetActors()
      a.InitTraversal()
      s=a.GetNextItem()
      while s:
          s.GetMapper().ScalarVisibilityOff()
          s=a.GetNextItem()              
                              
  def setColorSpace(self):
      self.cColorSpace.clear()
      self.cColorSpace.addItem("")
      self.cColorSpace.addItem("RGB")
      self.cColorSpace.addItem("HSV")
      self.cColorSpace.addItem("Diverging")
      QObject.connect(self.cColorSpace,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getColorSpace)

  def getColorSpace(self):
      index=self.cColorSpace.currentIndex()
      if (index==0): return
      colorspaces={1:self.lut.SetColorSpaceToRGB,2:self.lut.SetColorSpaceToHSV,
              3:self.lut.SetColorSpaceToDiverging}
      colorspaces[index]()
      
  def LookupTable(self):
      self.lut=vtk.vtkColorTransferFunction()
      self.lut.SetColorSpaceToRGB()
      self.lut.SetScaleToLinear()
      text=vtk.vtkTextProperty()
      text.SetFontFamilyToArial()
      text.SetFontSize(12)
      self.scalarbar=vtk.vtkScalarBarActor()
##       self.scalarbar.PickableOff()
      self.scalarbar.PickableOn()
      self.scalarbar.DragableOn()
      self.scalarbar.SetOrientationToVertical()
##       self.scalarbar.SetLabelTextProperty(text)
      self.scalarbar.SetLookupTable(self.lut)
      self.scalarbar.SetWidth(0.08)
      self.scalarbar.SetHeight(0.8)
      self.scalarbar.SetVisibility(0)
      self._vtkren.AddActor(self.scalarbar)
               
  def screenshot(self):
    sshot=QPixmap.grabWindow(self.display.winId())
    sshot.save('/tmp/foo.png','png')

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

  def findObjectDims(self,selected):
      return self._parser.getDimsFromObject(selected)
        
  def leave(self):
    self._wtop.destroy()
    self._control._hasvtkWindow=None
  
  def onexit(self):  
    self._control._control.delTreeView(self._viewid,
                                       self._fgprint.filedir,
                                       self._fgprint.filename)
    self.leave()     

  def addAxis(self):
    self.camAxes = vtk.vtkCamera()
    self.camAxes.ParallelProjectionOn()

    self.renAxes = vtk.vtkRenderer()
    self.renAxes.InteractiveOff()
##     self.renAxes.InteractiveOn()
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

  def wire(self):
      actors=self._vtkren.GetActors() 
      actors.InitTraversal()
      actor=actors.GetNextItem()
      while actor:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().EdgeVisibilityOff()
        actor = actors.GetNextItem()
    
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
      if (self.wact==1):
          self.wire()
          self.wact=0
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
      self._vtkwin.SetNumberOfLayers(2)
      self._vtkren.SetLayer(0)
      self._waxs.SetLayer(1)

      self._vtk.GetRenderWindow().AddRenderer(self._vtkren)
      self._vtk.GetRenderWindow().AddRenderer(self._waxs)
      
      self._parser=Mesh(T,zlist)
      self._selected=[]
      alist=self._parser.createActors()
      variables=[]
      if (alist!=[]):
          grid=vtk.vtkStructuredGrid.SafeDownCast(alist[0][2])
          if grid:
             for i in range(grid.GetCellData().GetNumberOfArrays()):
                 variables+=[grid.GetCellData().GetArray(i).GetName()]
          for a in alist:
              self._vtkren.AddActor(a[0])
              if (a[1] is not None):
                if (self._xmin>a[1][0]):self._xmin=a[1][0]
                if (self._ymin>a[1][2]):self._ymin=a[1][2]
                if (self._zmin>a[1][4]):self._zmin=a[1][4]
                if (self._xmax<a[1][1]):self._xmax=a[1][1]
                if (self._ymax<a[1][3]):self._ymax=a[1][3]
                if (self._zmax<a[1][5]):self._zmax=a[1][5]
      self.setVariable(variables)
      self._vtkren.SetBackground(1,1,1)
##       self._vtkren.ResetCamera()
      self._vtkren.GetActiveCamera().ParallelProjectionOn()
##       self._vtkren.GetActiveCamera().Elevation(0.0)
##       self._vtkren.GetActiveCamera().Azimuth(90.0)
##       self._vtkren.GetActiveCamera().Zoom(1.0)
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
      self.LookupTable()
      self._istyle=Q7InteractorStyle(self)
      self._vtk.SetInteractorStyle(self._istyle)
      self._iren=self._istyle.GetInteractor()
      self.fillCurrentPath()
      
      self._vtkren.AddObserver("StartEvent", self.SyncCameras)
      
      self._p_wire=True
      self.setColors(True)

      self._bindings={ 's'     :self.b_surf,
                       'S'     :self.b_surf,
                       'q'     :self.b_surfwire,
                       'Q'     :self.b_surfwire,
                       'a'     :self.wireActor,
                       'A'     :self.wireActor,
                       'w'     :self.b_wire,
                       'W'     :self.b_wire,
                       'r'     :self.resetCam,
                       'R'     :self.resetCam,
                       'd'     :self.hideActor,
                       'D'     :self.hideActor }
      
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
          actor.GetProperty().EdgeVisibilityOff()
          actor = actors.GetNextItem()
      self._vtk.GetRenderWindow().Render()

  def b_surfwire(self,pos):       
      if (not self._p_wire):
          self.b_wire(pos)
          return
      self._p_wire=False
      actors = self._vtkren.GetActors()
      actors.InitTraversal()
      actor = actors.GetNextItem()
      while actor:
          actor.GetProperty().SetRepresentationToSurface()
          actor.GetProperty().EdgeVisibilityOn()
          actor = actors.GetNextItem()
      if (not self.OutlineActor is None):
          self.OutlineActor.GetProperty().EdgeVisibilityOff()
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

  def wireActor(self,pos):
      if (self._selected==[]): return
      self.wire()
      for i in self._selected:
          i[1].GetProperty().SetRepresentationToSurface()
          i[1].GetProperty().EdgeVisibilityOff()
      self._vtk.GetRenderWindow().Render()
      self.wact=1

  def hideActor(self,pos):
      if (not self._currentactor is None):
          if (self._currentactor[0:2] in self._selected):
              self._currentactor[1].VisibilityOff()
              self._currentactor[3].VisibilityOff()
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
      self.resetSpinBox()
      if (not self.actorpt is None):
          self._vtkren.RemoveActor(self.actorpt)
      path =atp[0]
      actor=atp[1]
      if (not combo):self.setCurrentPath(path)         
      if (not self._currentactor is None):
          act=self._currentactor[3]
          col=self._currentactor[2]
          self._vtkren.RemoveActor(act)
          act.GetProperty().SetColor(col)
      if (actor is None):
          self.cCurrentPath.setCurrentIndex(0)
          if (not self.PickedRenderer is None and not self.OutlineActor is None):
              self.PickedRenderer.RemoveActor(self.OutlineActor)
              self.PickedRenderer=None
              self.CurrentRenderer.Render()
              self.CurrentRenderer.GetRenderWindow().Render()
          return
      self.grid_dims=self.findObjectDims(actor.GetMapper().GetInput())
##       actor.GetMapper().ScalarVisibilityOff()
      color=actor.GetProperty().GetColor()
      actor2=vtk.vtkActor()
      actor2.ShallowCopy(actor)
      actor2.PickableOff()
      actor2.DragableOff()
      actor2.GetProperty().SetColor(1,0,0)
      self._vtkren.AddActor(actor2)
      self.CurrentRenderer=self._vtkren
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
      self._currentactor=[path,actor,color,actor2]

  def highlightPoint(self,*args):
      actor=self._currentactor[1]
      if (actor is None): return
      grid=actor.GetMapper().GetInput()
      tgrid=self.grid_dims[0]
      if (tgrid==0):
          i=self.sIndex1.value()
          j=self.sIndex2.value()
          k=self.sIndex3.value()
          self.s_highlightPoint(grid,(i,j,k))
      if (tgrid==2):
          i=self.sIndex1.value()
          self.uns_highlightPoint(grid,i)
      if (tgrid==1):
          ptid=self.sIndex1.value()-1
          self.selectionPointId(grid,ptid)
      
  def s_highlightPoint(self,grid,index):
      if (not self.actorpt is None):
          self._vtkren.RemoveActor(self.actorpt)
      if (grid is None): return
      (i,j,k)=index
      filter=vtk.vtkStructuredGridGeometryFilter()
      filter.SetInputConnection(grid.GetProducerPort())
      filter.SetExtent(i-1,i-1,j-1,j-1,k-1,k-1)
      mapper=vtk.vtkPolyDataMapper()
      mapper.SetInputConnection(filter.GetOutputPort())
      self.actorpt=vtk.vtkActor()
      self.actorpt.SetMapper(mapper)
      self.actorpt.GetProperty().SetColor(0,1,0)
      self.actorpt.GetProperty().SetPointSize(6)
      self.actorpt.PickableOff()
      self.actorpt.DragableOff()
      self._vtkren.AddActor(self.actorpt)
      self._iren.Render()

  def uns_highlightPoint(self,grid,cellid):
      if (not self.actorpt is None):
          self._vtkren.RemoveActor(self.actorpt)
      if (grid is None): return
      filter=vtk.vtkUnstructuredGridGeometryFilter()
      filter.SetInputConnection(grid.GetProducerPort())
      filter.CellClippingOn()
      filter.SetCellMinimum(cellid)
      filter.SetCellMaximum(cellid)
      mapper=vtk.vtkDataSetMapper()
      mapper.SetInputConnection(filter.GetOutputPort())
      self.actorpt=vtk.vtkActor()
      self.actorpt.SetMapper(mapper)
      self.actorpt.GetProperty().SetColor(0,1,0)
      self.actorpt.GetProperty().SetLineWidth(2)
      self.actorpt.GetProperty().SetRepresentationToWireframe()
      self.actorpt.PickableOff()
      self.actorpt.DragableOff()
      self._vtkren.AddActor(self.actorpt)
      self._iren.Render()
      
  def selectionPointId(self,grid,ptid):
      if (not self.actorpt is None):
          self._vtkren.RemoveActor(self.actorpt)
      if (grid is None): return      
      ids=vtk.vtkIdTypeArray()
      ids.SetNumberOfComponents(1)
      ids.InsertNextValue(ptid)
      selectionNode = vtk.vtkSelectionNode()
      selectionNode.SetFieldType(1)
      selectionNode.SetContentType(4)
      selectionNode.SetSelectionList(ids)      
      selection=vtk.vtkSelection()
      selection.AddNode(selectionNode) 
      extractSelection = vtk.vtkExtractSelection()
      extractSelection.SetInputConnection(0,grid.GetProducerPort())
      extractSelection.SetInput(1,selection)
      extractSelection.Update()
      selected=vtk.vtkUnstructuredGrid()
      selected.ShallowCopy(extractSelection.GetOutput())
      selectedMapper=vtk.vtkDataSetMapper()
      selectedMapper.SetInputConnection(selected.GetProducerPort())
      self.actorpt=vtk.vtkActor()
      self.actorpt.SetMapper(selectedMapper)
      self.actorpt.GetProperty().SetColor(0,1,0)
      self.actorpt.GetProperty().SetPointSize(6)
      self.actorpt.PickableOff()
      self.actorpt.DragableOff()
      self._vtkren.AddActor(self.actorpt)      
      self._iren.Render()
      
  def setIndexPoint3(self,grid,index):
      (i,j,k)=index
      (idim,jdim,kdim)=self.grid_dims[1]
      self.sIndex1.blockSignals(True)
      self.sIndex2.blockSignals(True)
      self.sIndex3.blockSignals(True)
      self.sIndex1.setRange(1,idim)
      self.sIndex1.setSingleStep(1)
      self.sIndex2.setRange(1,jdim)
      self.sIndex2.setSingleStep(1)
      self.sIndex3.setRange(1,kdim)
      self.sIndex3.setSingleStep(1)
      self.sIndex1.setValue(i)
      self.sIndex2.setValue(j)
      self.sIndex3.setValue(k)
      self.sIndex1.blockSignals(False)
      self.sIndex2.blockSignals(False)
      self.sIndex3.blockSignals(False)
      self.s_highlightPoint(grid,index)
      
  def setIndexCell1(self,grid,index):
      idmax=grid.GetNumberOfCells()
      self.sIndex1.blockSignals(True)
      self.sIndex1.setRange(1,idmax)
      self.sIndex1.setSingleStep(1)
      self.sIndex1.setValue(index)
      self.sIndex1.blockSignals(False)
      self.uns_highlightPoint(grid,index)

  def setIndexPoint1(self,grid,index):
      idmax=grid.GetNumberOfPoints()
      self.sIndex1.blockSignals(True)
      self.sIndex1.setRange(1,idmax)
      self.sIndex1.setSingleStep(1)
      self.sIndex1.setValue(index+1)
      self.sIndex1.blockSignals(False)
      self.selectionPointId(grid,index)
      
  def setPickableOff(self):
      actors=self._vtkren.GetActors()
      actors.InitTraversal()
      a=actors.GetNextActor()
      a.PickableOff()
      while a:
          a.PickableOff()
          a=actors.GetNextActor()
      self._currentactor[1].PickableOn()

  def setPickableOn(self):
      actors=self._vtkren.GetActors()
      actors.InitTraversal()
      a=actors.GetNextActor()
      a.PickableOn()
      while a:
          a.PickableOn()
          a=actors.GetNextActor()
      if (not self._currentactor is None):
          self._currentactor[3].PickableOff()

  def resetSpinBox(self):  
      self.sIndex1.blockSignals(True)
      self.sIndex2.blockSignals(True)
      self.sIndex3.blockSignals(True)
      self.sIndex1.setRange(0,0)
      self.sIndex2.setRange(0,0)
      self.sIndex3.setRange(0,0)    
      self.sIndex1.setValue(0)
      self.sIndex2.setValue(0)
      self.sIndex3.setValue(0)
      self.sIndex1.blockSignals(False)
      self.sIndex2.blockSignals(False)
      self.sIndex3.blockSignals(False)

  def removeElement(self):
      if (self.actorpt is None): return
      self._vtkren.RemoveActor(self.actorpt)
      self._iren.Render()
      self.resetSpinBox()  
                 
class Q7InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self,parent):
      self._parent=parent
      self.AddObserver("CharEvent",self.keycode)
      self.rwi=None    
      self.PickedRenderer=None
      self.OutlineActor=None
      self._parent.addPicker()
        
    def keycode(self,*args):
        self.rwi=self.GetInteractor() 
        keycode=self.rwi.GetKeyCode()
        control=self.rwi.GetControlKey()        
        vtkkeys=['f','F','r','R']
        keys=['d','D','s','S','w','W','q','Q','a','A']
        if (keycode in vtkkeys):
            self.OnChar()
        if (keycode in keys):
            self.CharCallback()
        if (keycode=='z' or keycode=='Z'):
            self.setPick()
        if (keycode=='p' or keycode=='P'):
            self.pickElement()
        if (control==1): self._parent.controlKey=1

    def pickElement(self):
        if (self._parent._selected==[]): return
        grid=self._parent._currentactor[1].GetMapper().GetInput()
        tgrid=self._parent.grid_dims[0]
        eventPos=self.rwi.GetEventPosition()
        if (tgrid==0):
            pointid=self.getPointId(eventPos)
            if (pointid>-1):
                array=grid.GetPointData().GetArray(0).GetTuple3(pointid)
                self._parent.setIndexPoint3(grid,array)
            else:
                self._parent.removeElement()
        if (tgrid==2):
            cellid=self.getCellId(eventPos)
            if (cellid>-1):
                self._parent.setIndexCell1(grid,cellid)
            else:
                self._parent.removeElement()
        if (tgrid==1):
            pointid=self.getPointId(eventPos)
            if (pointid>-1):
                self._parent.setIndexPoint1(grid,pointid)
            else:
                self._parent.removeElement()
                           
    def getPointId(self,eventPos):
        picker=vtk.vtkPointPicker()
        picker.SetTolerance(0.005)            
        self.rwi.SetPicker(picker)
        self._parent.setPickableOff()
        picker.Pick(eventPos[0],eventPos[1], 0.0,self._parent._vtkren)
        pointid=picker.GetPointId()
        return pointid
        
    def getCellId(self,eventPos):
        picker=vtk.vtkCellPicker()
        picker.SetTolerance(0.001) 
        self.rwi.SetPicker(picker)
        self._parent.setPickableOff()
        picker.Pick(eventPos[0],eventPos[1], 0.0,self._parent._vtkren)
        cellid=picker.GetCellId()
        return cellid       
                       
    def setPick(self):
        self._parent.setPickableOn()
        path=None
        eventPos=self.rwi.GetEventPosition()
        self.rwi.SetPicker(self._parent.picker)
        if (not self._parent.picker is None):
          self._parent.picker.Pick(eventPos[0],eventPos[1],0.0,self._parent._vtkren)
          pathpick=self._parent.picker.GetPath()
          if (pathpick is None):
              self._parent.changeCurrentActor([None,None])
              self._parent.PropPicked=0
          else:
              actor=pathpick.GetFirstNode().GetViewProp()
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


        
