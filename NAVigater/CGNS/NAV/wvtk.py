#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

from PySide.QtCore    import *
from PySide.QtGui     import *
from CGNS.NAV.Q7VTKWindow import Ui_Q7VTKWindow
from CGNS.NAV.Q7VTKPlotWindow import Ui_Q7VTKPlotWindow
from CGNS.NAV.wfingerprint import Q7Window
try:
    from CGNS.PRO.vtkparser import Mesh
except:
    from CGNS.NAV.mparser import Mesh
from CGNS.NAV.moption import Q7OptionContext as OCTXT
from CGNS.NAV.wfile import Q7File

import CGNS.NAV.wmessages as MSG

import numpy as NPY

import math as M
import sys

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
  def __init__(self,control,parent,node,fgprint,tmodel,zlist):
      if (not zlist): pth='/'
      else: pth='<partial>'
      self._vtkstatus=False
      if (not self.wCGNSTreeParse(fgprint.tree,zlist)): return
      self._vtkstatus=True
      Q7Window.__init__(self,Q7Window.VIEW_VTK,control,pth,fgprint)
      self._xmin=self._ymin=self._zmin=self._xmax=self._ymax=self._zmax=0.0
      self._epix=QIcon(QPixmap(":/images/icons/empty.png"))
      self._spix=QIcon(QPixmap(":/images/icons/selected.png"))
      self._npix=QIcon(QPixmap(":/images/icons/unselected.png"))
      self._hpix=QIcon(QPixmap(":/images/icons/hidden.png"))
      self._T=self._fgprint.tree
      self._parent=parent
      self._master=parent
      self.lut=None
      self.grid_dims=None
      self.mincolor=None
      self.maxcolor=None
      self.scalarbar=None
      self.scalarbarwidget=None
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
      self.widgets=False
      self.resetSpinBox()
      self.display.Initialize()
      self.display.Start()
      self.display.show()
      self._vtktree=self.wCGNSTreeActors(fgprint.tree,zlist)
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
      self.selectable.clicked.connect(self.setInteractive)
      self.ColorMapMin=QColorDialog(self)
      self.ColorMapMax=QColorDialog(self)
      self.cCurrentPath.setParent(self)
      QObject.connect(self.ColorMapMin,SIGNAL("colorSelected(QColor)"),self.getColorMapMin)
      QObject.connect(self.ColorMapMax,SIGNAL("colorSelected(QColor)"),self.getColorMapMax)
      self.bZoom.clicked.connect(self.RubberbandZoom)
      self.bColorMapMin.clicked.connect(self.displayColorMapMin)
      self.bColorMapMax.clicked.connect(self.displayColorMapMax)
      self.bInfo.clicked.connect(self.infoVtkView)
      self.cShowValue.clicked.connect(self.showValues)
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
      QObject.connect(self.cShowZone, SIGNAL("stateChanged(int)"),
                      self.fillCurrentPath)
      QObject.connect(self.cShowBC, SIGNAL("stateChanged(int)"),
                      self.fillCurrentPath)
      QObject.connect(self.cShowMinMax, SIGNAL("stateChanged(int)"),
                      self.fillCurrentPath)
      QObject.connect(self.cRotationAxis,
                      SIGNAL("currentIndexChanged(int)"),self.setInteractor)
      QObject.connect(self.cVariables,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getVariable)
      QObject.connect(self.cColorSpace,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getColorSpace)
      self.bResetCamera.clicked.connect(self.ResetCam)

      self.Rotating=0
      self.Zooming=0
      self.Panning=0
      self.Rolling=0
      self.InteractorIndex=0
      self.txt=None
      self.observer=None
      self.actorval=None      
      self.planeActor=None

  @Slot(str)
  def printMessage(self, text):
        sys.stdout.write(text+'\n')
        sys.stdout.flush()

  def parseDone(self):
      print 'PARSE DONE'
      
  def setCutPlane(self):
      if (self._currentactor is None): return
      if (not vtk.vtkStructuredGrid().SafeDownCast(self._currentactor[1].GetMapper().GetInput())):
          return
      bounds=self._currentactor[1].GetBounds()
      bds=[0,0,0]
      bds[0]=(bounds[0]+bounds[1])/2.0
      bds[1]=(bounds[3]+bounds[2])/2.0
      bds[2]=(bounds[5]+bounds[4])/2.0
      grid=self._currentactor[1].GetMapper().GetInput()
      filter=vtk.vtkStructuredGridGeometryFilter()
      filter.SetInputConnection(grid.GetProducerPort())
      self.planeWidget.SetPlaceFactor(1.0)
      self.planeWidget.GetOutlineProperty().SetColor(0,1,1)
      self.planeWidget.OutlineTranslationOff()
      self.planeWidget.SetInput(filter.GetOutput())
      self.planeWidget.PlaceWidget()
      self.planeWidget.SetOrigin(bds[0],bds[1],bds[2])
      self.planeWidget.SetNormal(1,0,0)
      self.planeWidget.On()
      self.planeWidget.AddObserver("InteractionEvent",self.myCallback)
      self._iren.Render()

  def myCallback(self,*args):
      if (self.planeWidget.GetDrawPlane()==0):
          self.planeWidget.DrawPlaneOn()

  def cutting(self):
      if ((self._currentactor is None) or (self.planeWidget.GetEnabled()==0)): return
      if (self.planeActor is not None):
          self._vtkren.RemoveActor(self.planeActor)
      variable=self.cVariables.currentText()
      origin=self.planeWidget.GetOrigin()
      normal=self.planeWidget.GetNormal()
      plane=vtk.vtkPlane()
      plane.SetOrigin(origin)
      plane.SetNormal(normal)
      grid=self._currentactor[1].GetMapper().GetInput()
      filter=vtk.vtkStructuredGridGeometryFilter()
      filter.SetInputConnection(grid.GetProducerPort())
      cutter=vtk.vtkCutter()
      cutter.SetCutFunction(plane)
      cutter.SetInput(grid)
      cutter.Update()
      cutterMapper=vtk.vtkPolyDataMapper()
      cutterMapper.SetInputConnection(cutter.GetOutputPort())
      cutterMapper.SetScalarModeToUseCellData()
      cutterMapper.ScalarVisibilityOn()
      cutterMapper.SetLookupTable(self.lut)
      self.planeActor=vtk.vtkActor()
      self.planeActor.GetProperty().SetColor(1,1,0)
      self.planeActor.GetProperty().SetLineWidth(15)
      self.planeActor.GetProperty().SetRepresentationToSurface()
      self.planeActor.SetMapper(cutterMapper)
      self.planeActor.GetMapper().GetInput().GetCellData().SetActiveScalars(variable)
      self.planeWidget.DrawPlaneOff()
      self._vtkren.AddActor(self.planeActor)
      self._iren.Render()
            
  def infoVtkView(self):
      self._control.helpWindow('VTK')
      
  def mouseReleaseEvent(self, ev):
        self._ActiveButton=ev.button()
        ctrl=self._iren.GetControlKey()
        shift=self._iren.GetShiftKey()
        self._iren.SetEventInformationFlipY(ev.x(),ev.y(),
                                            ctrl,shift,chr(0),0,None)
        if (ev.button()==Qt.LeftButton):
            self._iren.LeftButtonReleaseEvent()
            self.Rotating=0
            self.Panning=0
            self.Rolling=0
        elif (ev.button()==Qt.RightButton):
            self._iren.RightButtonReleaseEvent()
            self.Zooming=0
        elif (ev.button()==Qt.MidButton):
            self._iren.MiddleButtonReleaseEvent()
            self.Panning=0

  def setInteractor(self,*args):
      if (self.cRotationAxis.currentIndex()==0):
          self._vtk.SetInteractorStyle(Q7InteractorStyle(self))
      elif (self.InteractorIndex==0):
          self._vtk.SetInteractorStyle(Q7InteractorStyleTrackballObject(self))
      self.InteractorIndex=self.cRotationAxis.currentIndex()


  def setCameraMode(self):
      self.cRotationAxis.clear()
      self.cRotationAxis.addItem("Camera axis all")
      self.cRotationAxis.addItem("Object axis all")
      self.cRotationAxis.addItem("Object axis x")    
      self.cRotationAxis.addItem("Object axis y")
      self.cRotationAxis.addItem("Object axis z")

  def ResetCam(self,*args):
      if (self.OutlineActor is None):
          self._vtkren.ResetCamera()
      else:
          self._vtkren.ResetCamera(self.OutlineActor.GetBounds())
      self._iren.Render()                                       
                  
  def RubberbandZoom(self,*args):
    if (self.bZoom.isChecked()):
      self._vtk.SetInteractorStyle(Q7InteractorStyleRubberBandZoom(self))
    else:
      interactor=self.cRotationAxis.currentIndex()
      if (interactor==0):
        self._vtk.SetInteractorStyle(Q7InteractorStyle(self))
      else:
        self._vtk.SetInteractorStyle(Q7InteractorStyleTrackballObject(self))

  def showValues(self,*args):
      if (self._currentactor is not None):
          if (not vtk.vtkStructuredGrid().SafeDownCast(self._currentactor[1].GetMapper().GetInput())):
              self.cShowValue.setChecked(0)
              return
      if ((self.cVariables.currentIndex()==0) or self.OutlineActor is None):
          self.cShowValue.setChecked(0)
          return
      if (self.cShowValue.isChecked()):
        self.observer=self._iren.AddObserver("MouseMoveEvent",self.displayScalars)
      elif (self.observer is not None):
        self._iren.RemoveObserver(self.observer)
          
  def displayScalars(self,*args):
      self.setPickableOff()
      eventPos=self._iren.GetEventPosition()
      picker=vtk.vtkCellPicker()
      picker.Pick(eventPos[0],eventPos[1],0.0,self._vtkren)
      cell=picker.GetCellId()
      if (cell>-1):
          array=self.cVariables.currentIndex()-1
          grid=self._currentactor[1].GetMapper().GetInput()
          data=grid.GetCellData().GetArray(array)
          value=data.GetTuple1(cell)
          value="%.3f" % value
          txt = vtk.vtkTextActor()
          txt.SetInput(value)
          txtprop=txt.GetTextProperty()
          txtprop.SetFontFamilyToArial()
          txtprop.SetFontSize(18)
          txtprop.SetColor(self.color)
          txt.SetDisplayPosition(eventPos[0],eventPos[1])
          self.selectionCellId(grid,cell)
          self._vtkren.AddActor(txt)
          if (self.txt is not None):
              self._vtkren.RemoveActor(self.txt)
              self._iren.Render()
          self.txt=txt
      else:
          if (self.txt is not None):
              self._vtkren.RemoveActor(self.txt)
          self._vtkren.RemoveActor(self.actorpt)
          self._iren.Render()
      self.setPickableOn()              
              
  def setInteractive(self,*args):
      if (self.selectable.isChecked()):
          if (not self.scalarbarwidget is None):
              self.scalarbarwidget.RepositionableOn()
              self.scalarbarwidget.ResizableOn()
          self.widget.InteractiveOn()

      else:
          if (not self.scalarbarwidget is None):
             self.scalarbarwidget.RepositionableOff()
             self.scalarbarwidget.ResizableOff()
          self.widget.InteractiveOff()
          
  def displayColorMapMin(self,*args):
      self.ColorMapMin.show()

  def displayColorMapMax(self,*args):
      self.ColorMapMax.show()

  def getColorMapMin(self,*args):
      self.maxcolor=self.ColorMapMin.currentColor().getRgbF()
      self.getVariable()

  def getColorMapMax(self,*args):
      self.mincolor=self.ColorMapMax.currentColor().getRgbF()
      self.getVariable()

  def setVariable(self,variable=[]):
      self.cVariables.clear()
      self.cVariables.addItem("")
      if (variable!=[]):
          for i in variable:
              self.cVariables.addItem(i)
      
  def getVariable(self,*args):
      if (self.planeWidget.GetEnabled()==1):
          self.planeWidget.Off()
      index=self.cVariables.currentIndex()
      if ((index==0) or (not self.cShowValue.isChecked())):
          if (self.observer is not None):
              self._iren.RemoveObserver(self.observer)
          self.cShowValue.setChecked(0)
      if ((self.mincolor is None) or (self.maxcolor is None)): return          
      variable=self.cVariables.currentText()
      name=variable[0:10]
      self.scalarbar.SetTitle(name)
      if (index==0):
          self.setVisibilityGrids(1)
          if (not self.scalarbarwidget is None):
              self.scalarbarwidget.Off()
              self.setScalarVisibilityOff()
              self.scalarbarwidget.GetScalarBarRepresentation().SetPosition(0.9,0.4)
              self.cShowValue.setChecked(0)
      else :
          a=self._vtkren.GetActors()
          a.InitTraversal()
          s=a.GetNextItem()
          minrange=[]
          maxrange=[]
          while s:
              grid=s.GetMapper().GetInput()
              tgrid=self.findObjectDims(grid)[0]
              if (tgrid==0):                        
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
              if (tgrid==1):
                  s.VisibilityOff()               
              s=a.GetNextItem()
          if (self.scalarbarwidget.GetEnabled()==0):
              sz=self._iren.GetRenderWindow().GetSize()
              self.scalarbarwidget.GetScalarBarRepresentation().SetPosition2(0.07*781/sz[0],0.6*554/sz[1])
              self.scalarbarwidget.On()
      if (self.OutlineActor is not None):
          self.OutlineActor.VisibilityOn()
      self._iren.Render()

  def setVisibilityGrids(self,n):
      a=self._vtkren.GetActors()
      a.InitTraversal()
      s=a.GetNextItem()
      while s:
          grid=s.GetMapper().GetInput()
          if (vtk.vtkUnstructuredGrid.SafeDownCast(grid)):
              s.SetVisibility(n)
          s=a.GetNextItem()      
            
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

  def getColorSpace(self):
      index=self.cColorSpace.currentIndex()
      if (index==0): return
      colorspaces={1:self.lut.SetColorSpaceToRGB,2:self.lut.SetColorSpaceToHSV,
              3:self.lut.SetColorSpaceToDiverging}
      colorspaces[index]()
      self._iren.Render()
      
  def LookupTable(self):
      self.lut=vtk.vtkColorTransferFunction()
      self.lut.SetColorSpaceToRGB()
      self.lut.SetScaleToLinear()
      text=vtk.vtkTextProperty()
      text.SetFontFamilyToArial()
      text.SetFontSize(12)
      text.SetColor(self.color)
      self.scalarbar=vtk.vtkScalarBarActor()
      self.scalarbar.SetLookupTable(self.lut)
      self.scalarbar.SetNumberOfLabels(5)
      self.scalarbar.SetLabelTextProperty(text)
      self.scalarbar.SetTitleTextProperty(text)
      self.scalarbarwidget=vtk.vtkScalarBarWidget()
      self.scalarbarwidget.ResizableOff()      
      self.scalarbarwidget.RepositionableOff()
      self.scalarbarwidget.GetScalarBarRepresentation().SetScalarBarActor(self.scalarbar)
      self.scalarbarwidget.GetScalarBarRepresentation().PickableOff()
      self.scalarbarwidget.GetScalarBarRepresentation().SetPosition(0.9,0.4)
      self.scalarbarwidget.GetScalarBarRepresentation().SetOrientation(1)
      
  def screenshot(self):
    sshot=QPixmap.grabWindow(self.display.winId())
    sshot.save('/tmp/foo.png','png')

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
        
#  def leave(self):
#    self._wtop.destroy()
#    print self._parent._vtkwindow
#    self._parent._vtkwindow=None
#  
#  def onexit(self):  
#    self._control._control.delTreeView(self._viewid,
#                                       self._fgprint.filedir,
#                                       self._fgprint.filename)
#    self.leave()     

  def addMarker(self):
    axes=vtk.vtkAxesActor()
    axes.SetShaftTypeToLine()
    axes.SetTotalLength(1,1,1)
    
    self.widget=vtk.vtkOrientationMarkerWidget()
    self.widget.SetOutlineColor(0.93,0.57,0.13)
    self.widget.SetOrientationMarker(axes)
    self.widget.SetViewport(0,0,0.25,0.25)
    
    xLabel=axes.GetXAxisCaptionActor2D()
    xLabel.SetCaption("X")
    xprop=vtk.vtkTextProperty()
    xprop.SetFontSize(3)
    xLabel.SetCaptionTextProperty(xprop)
    xLabel.SetAttachmentPoint(0.75,0.2,0)
    xLabel.LeaderOff()
    xLabel.BorderOff()
    xLabel.GetProperty().SetColor(self.color)
    xLabel.SetPosition(0,0)

    yLabel=axes.GetYAxisCaptionActor2D()
    yLabel.SetCaption("Y")
    yprop=vtk.vtkTextProperty()
    yprop.SetFontSize(3)
    yLabel.SetCaptionTextProperty(yprop)
    yLabel.SetAttachmentPoint(0.2,0.75,0)
    yLabel.LeaderOff()
    yLabel.BorderOff()
    yLabel.GetProperty().SetColor(self.color)
    yLabel.SetPosition(0,0)

    zLabel=axes.GetZAxisCaptionActor2D() 
    zLabel.SetCaption("Z")
    zprop=vtk.vtkTextProperty()
    zprop.SetFontSize(3)
    zLabel.SetCaptionTextProperty(zprop)
    zLabel.SetAttachmentPoint(0,0.2,0.75)
    zLabel.LeaderOff()
    zLabel.BorderOff()
    zLabel.GetProperty().SetColor(self.color)
    zLabel.SetPosition(0,0)

    self.xLabel=xLabel
    self.yLabel=yLabel
    self.zLabel=zLabel
    
  def colourLabel(self):
      self.xLabel.GetProperty().SetColor(self.color)
      self.yLabel.GetProperty().SetColor(self.color)
      self.zLabel.GetProperty().SetColor(self.color)
      self.scalarbar.GetLabelTextProperty().SetColor(self.color)
      self.scalarbar.GetTitleTextProperty().SetColor(self.color)
      self.textwidget.GetTextActor().GetTextProperty().SetColor(self.color)
      
  def wire(self):
      actors=self._vtkren.GetActors() 
      actors.InitTraversal()
      actor=actors.GetNextItem()
      while actor:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().EdgeVisibilityOff()
        actor=actors.GetNextItem()
    
  def addPicker(self):
      txtActor=vtk.vtkActor2D()
      self.txtMapper=vtk.vtkTextMapper()
      txtActor.SetMapper(self.txtMapper)
      self.textActor=vtk.vtkTextActor()
      self.textActor.VisibilityOff()
      self.textActor.PickableOff()
      tprop=vtk.vtkTextProperty()
      tprop.SetFontFamilyToArial()
      tprop.SetFontSize(10)
      tprop.BoldOff()
      tprop.ShadowOff()
      tprop.SetColor(self.color)
      self.textActor.SetTextProperty(tprop)
      self.picker = vtk.vtkCellPicker()
      self.picker.SetTolerance(0.001)
      self.picker.AddObserver("EndPickEvent",self.annotatePick)
      self.textrepresentation=vtk.vtkTextRepresentation()
      self.textrepresentation.GetPositionCoordinate().SetCoordinateSystem(3)
      self.textwidget=vtk.vtkTextWidget()
      self.textwidget.SetRepresentation(self.textrepresentation)
      self.textwidget.SetTextActor(self.textActor)
      self.textwidget.SelectableOff()
      self.textwidget.ResizableOff()

  def annotatePick(self,object,event):
      if (self.wact==1):
          self.wire()
          self.wact=0
      if (self.controlKey!=1):
        self._selected=[]
        self.PropPicked=0
      if (self.picker.GetCellId()<0):
        self.textActor.VisibilityOff()
        self.textwidget.Off()
        self.highlightProp=0
      else:                  
        pickAct = self.picker.GetActors()
        pickAct.InitTraversal()
        a=pickAct.GetNextItem()        
        t=''
        s=None
        while a:
          x=a.GetMapper().GetInput()
          s=self.findObjectPath(x)
          t+=s+'\n'
          self._selected+=[[s,a]]
          a=pickAct.GetNextItem()
        self.txtMapper.SetInput(t)
        sz=self._iren.GetRenderWindow().GetSize()[1]
        y=sz-self.txtMapper.GetHeight(self._vtkren)-10.
        h=y/sz
        self.textActor.VisibilityOn()
        self.textActor.SetInput(t)
        self.textActor.SetTextScaleModeToNone()
        self.textActor.GetTextProperty().SetJustificationToLeft()
        self.textrepresentation.SetPosition(0.02,h)
        self.textwidget.On()
        self.PropPicked=0
        self.controlKey=0
        self.highlightProp=1
        
  def wCGNSTreeParse(self,T,zlist):
      self._parser=Mesh(T,zlist)
      if (not self._parser._status):
          MSG.wError(823,'Cannot create VTK view',
                     'No data or bad data for a VTK display')
          return False
      return True

  def wCGNSTreeActors(self,T,zlist):
      o=vtk.vtkObject()
      o.SetGlobalWarningDisplay(0)
      del o
      
      self._vtk=self.display
      self._vtk.setParent(self)
      self._vtkren = vtk.vtkRenderer()
      
      self.addMarker()
      self._vtkwin=self._vtk.GetRenderWindow()
      self._vtkwin.SetNumberOfLayers(1)

      self._vtk.GetRenderWindow().AddRenderer(self._vtkren)
      
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
      self._vtkren.GetActiveCamera().ParallelProjectionOn()
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
      self.addPicker()
      self._istyle=Q7InteractorStyle(self)
      self._vtk.SetInteractorStyle(self._istyle)
      self._iren=self._istyle.GetInteractor()
      self.scalarbarwidget.SetInteractor(self._iren)
      self.textwidget.SetInteractor(self._iren)
      self.widget.SetInteractor(self._iren)
      self.widget.On()
      self.widget.InteractiveOff()
      self.planeWidget=vtk.vtkImplicitPlaneWidget()
      self.planeWidget.SetInteractor(self._iren)
      self.fillCurrentPath()
      
      self._vtkren.AddObserver("StartEvent", self.posWidgets)
      self.setCameraMode()
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
                       'u'     :self.b_reset,
                       'U'     :self.b_reset,
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
      if (self.actorpt is not None):
          self.actorpt.GetProperty().SetColor(0,1,0)
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
      
  def b_reset(self,*args):
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
      pthlist=[]
      if (self.cShowZone.isChecked()): pthlist+=['Zone']
      if (self.cShowMinMax.isChecked()): pthlist+=['Min/Max']
      if (self.cShowBC.isChecked()): pthlist+=['BC']
      for i in self._parser.getPathList(pthlist):
          pix=self._npix
          if (i in sel): pix=self._spix
          if (i in hid): pix=self._hpix
          self.cCurrentPath.addItem(pix,i)
      self.cCurrentPath.model().sort(0)
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
     
  def posWidgets(self,*args):
      if (self._iren is not None):
          sz=self._iren.GetRenderWindow().GetSize()
          self.posScalarBar(sz)

  def posScalarBar(self,sz):
      if (self.scalarbarwidget.GetEnabled()==1):
          if (self.scalarbarwidget.GetScalarBarActor().GetOrientation()==1):
              self.scalarbarwidget.GetScalarBarRepresentation().SetPosition2(0.07*781/sz[0],0.6*554/sz[1])
          else:
              self.scalarbarwidget.GetScalarBarRepresentation().SetPosition2(0.6*781/sz[0],0.07*554/sz[1])
          pos2=self.scalarbarwidget.GetScalarBarRepresentation().GetPosition2()
          pos=self.scalarbarwidget.GetScalarBarRepresentation().GetPosition()
          pos=list(pos)
          if (pos[0]<=0):
              pos[0]=0          
          if (pos[1]<=0):
              pos[1]=0
          if ((pos2[1]+pos[1])>=1):
              pos[1]=1-pos2[1]
              self.scalarbarwidget.GetScalarBarRepresentation().SetPosition(pos[0],1.2-pos2[1])
          if ((pos2[0]+pos[0])>=1.01):
              pos[0]=1.01-pos2[0]  
          self.scalarbarwidget.GetScalarBarRepresentation().SetPosition(pos[0],pos[1])
                 
  def b_surfwire(self,pos):       
      if (not self._p_wire):
          self.b_wire(pos)
          return
      self._p_wire=False
      actors=self._vtkren.GetActors()
      actors.InitTraversal()
      actor=actors.GetNextItem()
      while actor:
          actor.GetProperty().SetRepresentationToSurface()
          actor.GetProperty().EdgeVisibilityOn()
          actor=actors.GetNextItem()
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
        
  def reject(self):
      if (self._master._vtkwindow is not None):
          self._vtk.GetRenderWindow().Finalize()
          self._master._vtkwindow=None
      self.close()
        
  def changeCurrentActor(self,atp,combo=True):
      self.resetSpinBox()
      if (self.planeWidget is not None):
          if (self.planeWidget.GetEnabled()==1):
              self.planeWidget.Off()
          if (self.planeActor is not None):
              self._vtkren.RemoveActor(self.planeActor)
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
          self.OutlineActor.SetScale(1.01,1.01,1.01)
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

  def selectionCellId(self,grid,ptid):
      if (not self.actorpt is None):
          self._vtkren.RemoveActor(self.actorpt)
      if (grid is None): return      
      ids=vtk.vtkIdTypeArray()
      ids.SetNumberOfComponents(1)
      ids.InsertNextValue(ptid)
      selectionNode = vtk.vtkSelectionNode()
      selectionNode.SetFieldType(0)
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

  def keycode(self,*args):
      keycode=self._iren.GetKeyCode()
      control=self._iren.GetControlKey()        
      vtkkeys=['f','F','r','R']
      keys=['d','D','s','S','w','W','q','Q','a','A','u','U']
      if (keycode in vtkkeys):
          self.OnChar()
      if (keycode in keys):
          self.CharCallback()
      if (keycode in ['z','Z',' ']):
          self.setPick()
      if (keycode=='p' or keycode=='P'):
          self.pickElement()
      if (control==1): self.controlKey=1

  def pickElement(self):
      if ((self._selected==[]) or self.cShowValue.isChecked()): return
      grid=self._currentactor[1].GetMapper().GetInput()
      tgrid=self.grid_dims[0]
      eventPos=self._iren.GetEventPosition()
      if (tgrid==0):
          pointid=self.getPointId(eventPos)
          if (pointid>-1):
              array=grid.GetPointData().GetArray(0).GetTuple3(pointid)
              self.setIndexPoint3(grid,array)
          else:
              self.removeElement()
      if (tgrid==2):
          cellid=self.getCellId(eventPos)
          if (cellid>-1):
              self.setIndexCell1(grid,cellid)
          else:
              self.removeElement()
      if (tgrid==1):
          pointid=self.getPointId(eventPos)
          if (pointid>-1):
              self.setIndexPoint1(grid,pointid)
          else:
              self.removeElement()

  def setPick(self):
      self.setPickableOn()
      path=None
      eventPos=self._iren.GetEventPosition()
      self._iren.SetPicker(self.picker)
      if (not self.picker is None):
          self.picker.Pick(eventPos[0],eventPos[1],0.0,self._vtkren)
          pathpick=self.picker.GetPath()
          if (pathpick is None):
              self.changeCurrentActor([None,None])
              self.OutlineActor=None
              self.PropPicked=0
          else:
              actor=pathpick.GetFirstNode().GetViewProp()
              path=self.findObjectPath(actor.GetMapper().GetInput())
              self.changeCurrentActor([path,actor])
              self.PropPicked=1
          self.fillCurrentPath()
          if (path is not None):
            self.setCurrentPath(path)
                  
  def getPointId(self,eventPos):
      picker=vtk.vtkPointPicker()
      picker.SetTolerance(0.005)            
      self._iren.SetPicker(picker)
      self.setPickableOff()
      picker.Pick(eventPos[0],eventPos[1], 0.0,self._vtkren)
      pointid=picker.GetPointId()
      return pointid

  def getCellId(self,eventPos):
      picker=vtk.vtkCellPicker()
      picker.SetTolerance(0.001) 
      self._iren.SetPicker(picker)
      self.setPickableOff()
      picker.Pick(eventPos[0],eventPos[1], 0.0,self._vtkren)
      cellid=picker.GetCellId()
      return cellid       
                            
  def CharCallback(self,*args):   
      keycode=self._iren.GetKeyCode()
      control=self._iren.GetControlKey()
      pos=self._iren.GetEventPosition()
      if (self._bindings.has_key(keycode)): self._bindings[keycode](pos)
      if (control==1): self.controlKey=1     
      return

# -----------------------------------------------------------------------------
class Q7InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self,parent):
      self._parent=parent
      self.AddObserver("CharEvent",self.keycode)
      
    def keycode(self,*args):
      keycode=self._parent._iren.GetKeyCode()
      control=self._parent._iren.GetControlKey()
      vtkkeys=['f','F','r','R']
      keys=['d','D','s','S','w','W','q','Q','a','A','u','U']
      if (keycode in vtkkeys):
          self.OnChar()
      if (keycode in keys):
          self._parent.CharCallback()
      if (keycode in ['z','Z',' ']):
          self._parent.setPick()
      if (keycode=='p' or keycode=='P'):
          self._parent.pickElement()
      if (keycode=='o' or keycode=='O'):
          self._parent.setCutPlane()
      if (keycode=='t' or keycode=='T'):
          self._parent.cutting()      
      if (control==1): self._parent.controlKey=1

# -----------------------------------------------------------------------------
class Q7InteractorStyleTrackballObject(vtk.vtkInteractorStyle):
    def __init__(self,parent):
      self._parent=parent
      self.AddObserver("CharEvent",self.keycode)  
      self.AddObserver("LeftButtonPressEvent", self.ButtonEvent)
      self.AddObserver("MouseMoveEvent", self.MouseMove)
      self.AddObserver("RightButtonPressEvent", self.ButtonEvent)
      self.AddObserver("MiddleButtonPressEvent", self.ButtonEvent)
      self.AddObserver("MouseWheelForwardEvent", self.MouseWheelForward)
      self.AddObserver("MouseWheelBackwardEvent", self.MouseWheelBackward)

    def MouseWheelForward(self,*args):
        factor=pow(1.1,2.0)
        self.Zoom(factor)

    def MouseWheelBackward(self,*args):
        factor=pow(1.1,-2.0)
        self.Zoom(factor)
        
    def keycode(self,*args):
      keycode=self._parent._iren.GetKeyCode()
      control=self._parent._iren.GetControlKey()        
      vtkkeys=['f','F','r','R']
      keys=['d','D','s','S','w','W','q','Q','a','A','u','U']
      if (keycode in vtkkeys):
          self.OnChar()
      if (keycode in keys):
          self._parent.CharCallback()
      if (keycode in ['z','Z',' ']):
          self._parent.setPick()
      if (keycode=='p' or keycode=='P'):
          self._parent.pickElement()
      if (control==1): self._parent.controlKey=1

    def ButtonEvent(self,obj,event):
      if (event=="LeftButtonPressEvent"):
          if (self._parent._iren.GetShiftKey()):
              self._parent.Panning=1
          elif (self._parent._iren.GetControlKey()):
              self._parent.Rolling=1
          else:
              self._parent.Rotating=1
      if (event=="RightButtonPressEvent"):
          self._parent.Zooming=1
      if (event=="MiddleButtonPressEvent"):
          self._parent.Panning=1
          
    def MouseMove(self,obj,event):
      index=self._parent.cRotationAxis.currentIndex()
      axis={2:[1,0,0],3:[0,1,0],4:[0,0,1]}
      if self._parent.Rotating:
          if (index==1):
              self.CameraModeToObjectAllAxis()
          else:
              self.CameraModeToObjectAxis(axis[index])
      if self._parent.Zooming:
          self.Dolly()
      if self._parent.Panning:
          self.Pan()
      if self._parent.Rolling:
          self.Roll()

    def ComputeDisplayCenter(self):
      if (self._parent.OutlineActor is None):
          bounds=(self._parent._xmin,self._parent._xmax,self._parent._ymin,
                  self._parent._ymax,self._parent._zmin,self._parent._zmax)
      else:
          bounds=self._parent._currentactor[1].GetBounds()
      center=[0,0,0]
      center[0]=(bounds[0]+bounds[1])/2.0
      center[1]=(bounds[2]+bounds[3])/2.0
      center[2]=(bounds[4]+bounds[5])/2.0
      self._parent._vtkren.SetWorldPoint(center[0],center[1],center[2],1.0)
      self._parent._vtkren.WorldToDisplay()
      pt=self._parent._vtkren.GetDisplayPoint()
      return pt,center

    def CameraModeToObjectAxis(self,axis):
      camera=self._parent._vtkren.GetActiveCamera()
      transform=vtk.vtkTransform()
      transform.Identity()
      (pt,center)=self.ComputeDisplayCenter()     
      transform.Translate(center[0],center[1],center[2])
      dx=self._parent._iren.GetLastEventPosition()[0]-self._parent._iren.GetEventPosition()[0]
      dy=self._parent._iren.GetLastEventPosition()[1]-self._parent._iren.GetEventPosition()[1]
      camera.OrthogonalizeViewUp()
      size=self._parent._vtkren.GetSize()
      transform.RotateWXYZ(360.0*dx/size[0],axis[0],axis[1],axis[2])
      transform.RotateWXYZ(360.0*dy/size[1],axis[0],axis[1],axis[2])
      transform.Translate(-center[0],-center[1],-center[2])
      camera.ApplyTransform(transform)
      camera.OrthogonalizeViewUp()
      self._parent._vtkren.ResetCameraClippingRange()
      self._parent._vtkren.GetRenderWindow().Render()

    def CameraModeToObjectAllAxis(self):
      camera=self._parent._vtkren.GetActiveCamera()
      transform=vtk.vtkTransform()
      transform.Identity()
      (pt,center)=self.ComputeDisplayCenter()     
      transform.Translate(center[0],center[1],center[2])
      dx=self._parent._iren.GetLastEventPosition()[0]-self._parent._iren.GetEventPosition()[0]
      dy=self._parent._iren.GetLastEventPosition()[1]-self._parent._iren.GetEventPosition()[1]
      camera.OrthogonalizeViewUp()
      viewUp=camera.GetViewUp()
      size=self._parent._vtkren.GetSize()          
      transform.RotateWXYZ(360.0*dx/size[0],viewUp[0],viewUp[1],viewUp[2])
      v2=[0,0,0]
      vtk.vtkMath().Cross(camera.GetDirectionOfProjection(),viewUp,v2)
      transform.RotateWXYZ(-360.0*dy/size[1],v2[0],v2[1],v2[2])
      transform.Translate(-center[0],-center[1],-center[2])
      camera.ApplyTransform(transform)
      camera.OrthogonalizeViewUp()
      self._parent._vtkren.ResetCameraClippingRange()
      self._parent._vtkren.GetRenderWindow().Render()

    def Dolly(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        center=self._parent._vtkren.GetCenter()
        dy=self._parent._iren.GetEventPosition()[1]-self._parent._iren.GetLastEventPosition()[1]
        dyf=10*dy/center[1]
        dollyFactor=pow(1.1,dyf)
        self.Zoom(dollyFactor)

    def Zoom(self,factor):
        camera=self._parent._vtkren.GetActiveCamera()
        camera.SetParallelScale(camera.GetParallelScale()/factor)
        if self._parent._iren.GetLightFollowCamera():
            self._parent._vtkren.UpdateLightsGeometryToFollowCamera()
        self._parent._iren.Render()       
                                        
    def Pan(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        motionVector=[0,0,0]
        fp=camera.GetFocalPoint()
        
        self._parent._vtkren.SetWorldPoint(fp[0],fp[1],fp[2],1.0)
        self._parent._vtkren.WorldToDisplay()
        dp=self._parent._vtkren.GetDisplayPoint()
        focalDepth=dp[2]
        self._parent._vtkren.SetDisplayPoint(self._parent._iren.GetEventPosition()[0],
                                 self._parent._iren.GetEventPosition()[1],focalDepth)
        self._parent._vtkren.DisplayToWorld()
        newPickPoint=self._parent._vtkren.GetWorldPoint()
        self._parent._vtkren.SetDisplayPoint(self._parent._iren.GetLastEventPosition()[0],
                                 self._parent._iren.GetLastEventPosition()[1],focalDepth)
        self._parent._vtkren.DisplayToWorld()
        oldPickPoint=self._parent._vtkren.GetWorldPoint()
        motionVector[0]=oldPickPoint[0]-newPickPoint[0]
        motionVector[1]=oldPickPoint[1]-newPickPoint[1]
        motionVector[2]=oldPickPoint[2]-newPickPoint[2]
  
        viewFocus=list(camera.GetFocalPoint())
        viewPoint=list(camera.GetPosition())
        camera.SetFocalPoint(motionVector[0]+viewFocus[0],
                        motionVector[1]+viewFocus[1],
                        motionVector[2]+viewFocus[2])

        camera.SetPosition(motionVector[0]+viewPoint[0],
                      motionVector[1]+viewPoint[1],
                      motionVector[2]+viewPoint[2])
        if self._parent._iren.GetLightFollowCamera():
            self._parent._vtkren.UpdateLightsGeometryToFollowCamera()
        self._parent._iren.Render()                

    def Roll(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        transform=vtk.vtkTransform()
        pos=camera.GetPosition()
        fp=camera.GetFocalPoint()
        axis=[fp[i]-pos[i] for i in range(3)]
        (DisplayCenter,Center)=self.ComputeDisplayCenter()
        x1=self._parent._iren.GetLastEventPosition()[0]-int(DisplayCenter[0])
        x2=self._parent._iren.GetEventPosition()[0]-int(DisplayCenter[0])
        y1=self._parent._iren.GetLastEventPosition()[1]-int(DisplayCenter[1])
        y2=self._parent._iren.GetEventPosition()[1]-int(DisplayCenter[1])
        zCross=x1*y2-y1*x2
        zCross=float(zCross)
        angle=vtk.vtkMath().DegreesFromRadians(zCross/(M.sqrt(float(x1*x1+y1*y1))*
                                                     M.sqrt(float(x2*x2+y2*y2))))
        transform.Identity()
        transform.Translate(Center[0],Center[1],Center[2])
        transform.RotateWXYZ(angle,axis[0],axis[1],axis[2])
        transform.Translate(-Center[0],-Center[1],-Center[2])
        camera.ApplyTransform(transform)
        camera.OrthogonalizeViewUp()
        self._parent._vtkren.ResetCameraClippingRange()
        self._parent._iren.Render()                                            

# -----------------------------------------------------------------------------
class Q7InteractorStyleRubberBandZoom(vtk.vtkInteractorStyleRubberBandZoom):
    def __init__(self,parent):
      self._parent=parent
      self.AddObserver("CharEvent",self.keycode)  
      self.AddObserver("MouseMoveEvent", self.MouseMove)
      self.AddObserver("LeftButtonPressEvent", self.ButtonEvent)
      self.AddObserver("RightButtonPressEvent", self.ButtonEvent)
      self.AddObserver("MiddleButtonPressEvent", self.ButtonEvent)
      self.AddObserver("MouseWheelForwardEvent", self.MouseWheelForward)
      self.AddObserver("MouseWheelBackwardEvent", self.MouseWheelBackward)

    def MouseWheelForward(self,*args):
        factor=pow(1.1,2.0)
        self.Zoom(factor)

    def MouseWheelBackward(self,*args):
        factor=pow(1.1,-2.0)
        self.Zoom(factor)
        
    def keycode(self,*args):
      keycode=self._parent._iren.GetKeyCode()
      control=self._parent._iren.GetControlKey()        
      vtkkeys=['f','F','r','R']
      keys=['d','D','s','S','w','W','q','Q','a','A','u','U']
      if (keycode in vtkkeys):
          self.OnChar()
      if (keycode in keys):
          self._parent.CharCallback()
      if (keycode in ['z','Z',' ']):
          self._parent.setPick()
      if (keycode=='p' or keycode=='P'):
          self._parent.pickElement()
      if (control==1): self._parent.controlKey=1

    def ButtonEvent(self,obj,event):
      if (event=="LeftButtonPressEvent"):
          if (self._parent._iren.GetShiftKey()):
              self._parent.Panning=1
          elif (self._parent._iren.GetControlKey()):
              self._parent.Rolling=1
          else:
              self.OnLeftButtonDown()
      if (event=="RightButtonPressEvent"):
          self._parent.Zooming=1
      if (event=="MiddleButtonPressEvent"):
          self._parent.Panning=1
          
    def MouseMove(self,obj,event):
      index=self._parent.cRotationAxis.currentIndex()
      if self._parent.Zooming:
          self.Dolly()
      if self._parent.Panning:
          self.Pan()
      if self._parent.Rolling:
          self.Roll()
      self.OnMouseMove()

    def ComputeDisplayCenter(self):
      if (self._parent.OutlineActor is None):
          bounds=(self._parent._xmin,self._parent._xmax,self._parent._ymin,
                  self._parent._ymax,self._parent._zmin,self._parent._zmax)
      else:
          bounds=self._parent._currentactor[1].GetBounds()
      center=[0,0,0]
      center[0]=(bounds[0]+bounds[1])/2.0
      center[1]=(bounds[2]+bounds[3])/2.0
      center[2]=(bounds[4]+bounds[5])/2.0
      self._parent._vtkren.SetWorldPoint(center[0],center[1],center[2],1.0)
      self._parent._vtkren.WorldToDisplay()
      pt=self._parent._vtkren.GetDisplayPoint()
      return pt,center

    def Dolly(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        center=self._parent._vtkren.GetCenter()
        dy=self._parent._iren.GetEventPosition()[1]-self._parent._iren.GetLastEventPosition()[1]
        dyf=10*dy/center[1]
        dollyFactor=pow(1.1,dyf)
        self.Zoom(dollyFactor)

    def Zoom(self,factor):
        camera=self._parent._vtkren.GetActiveCamera()
        camera.SetParallelScale(camera.GetParallelScale()/factor)
        if self._parent._iren.GetLightFollowCamera():
            self._parent._vtkren.UpdateLightsGeometryToFollowCamera()
        self._parent._iren.Render()       
                                        
    def Pan(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        motionVector=[0,0,0]
        fp=camera.GetFocalPoint()
        
        self._parent._vtkren.SetWorldPoint(fp[0],fp[1],fp[2],1.0)
        self._parent._vtkren.WorldToDisplay()
        dp=self._parent._vtkren.GetDisplayPoint()
        focalDepth=dp[2]
        self._parent._vtkren.SetDisplayPoint(self._parent._iren.GetEventPosition()[0],
                                 self._parent._iren.GetEventPosition()[1],focalDepth)
        self._parent._vtkren.DisplayToWorld()
        newPickPoint=self._parent._vtkren.GetWorldPoint()
        self._parent._vtkren.SetDisplayPoint(self._parent._iren.GetLastEventPosition()[0],
                                 self._parent._iren.GetLastEventPosition()[1],focalDepth)
        self._parent._vtkren.DisplayToWorld()
        oldPickPoint=self._parent._vtkren.GetWorldPoint()
        motionVector[0]=oldPickPoint[0]-newPickPoint[0]
        motionVector[1]=oldPickPoint[1]-newPickPoint[1]
        motionVector[2]=oldPickPoint[2]-newPickPoint[2]
  
        viewFocus=list(camera.GetFocalPoint())
        viewPoint=list(camera.GetPosition())
        camera.SetFocalPoint(motionVector[0]+viewFocus[0],
                        motionVector[1]+viewFocus[1],
                        motionVector[2]+viewFocus[2])

        camera.SetPosition(motionVector[0]+viewPoint[0],
                      motionVector[1]+viewPoint[1],
                      motionVector[2]+viewPoint[2])
        if self._parent._iren.GetLightFollowCamera():
            self._parent._vtkren.UpdateLightsGeometryToFollowCamera()
        self._parent._iren.Render()                

    def Roll(self,*args):
        camera=self._parent._vtkren.GetActiveCamera()
        transform=vtk.vtkTransform()
        pos=camera.GetPosition()
        fp=camera.GetFocalPoint()
        axis=[fp[i]-pos[i] for i in range(3)]
        (DisplayCenter,Center)=self.ComputeDisplayCenter()
        x1=self._parent._iren.GetLastEventPosition()[0]-int(DisplayCenter[0])
        x2=self._parent._iren.GetEventPosition()[0]-int(DisplayCenter[0])
        y1=self._parent._iren.GetLastEventPosition()[1]-int(DisplayCenter[1])
        y2=self._parent._iren.GetEventPosition()[1]-int(DisplayCenter[1])
        zCross=x1*y2-y1*x2
        zCross=float(zCross)
        angle=vtk.vtkMath().DegreesFromRadians(zCross/(M.sqrt(float(x1*x1+y1*y1))*
                                                     M.sqrt(float(x2*x2+y2*y2))))
        transform.Identity()
        transform.Translate(Center[0],Center[1],Center[2])
        transform.RotateWXYZ(angle,axis[0],axis[1],axis[2])
        transform.Translate(-Center[0],-Center[1],-Center[2])
        camera.ApplyTransform(transform)
        camera.OrthogonalizeViewUp()
        self._parent._vtkren.ResetCameraClippingRange()
        self._parent._iren.Render()                                 
    
# -----------------------------------------------------------------
class Q7VTKPlot(Q7Window,Ui_Q7VTKPlotWindow):
  def __init__(self,control,node,fgprint,tmodel,zlist):   
      if (not zlist): pth='/'
      else: pth='<partial>'
      self._vtkstatus=False
      Q7Window.__init__(self,Q7Window.VIEW_VTK,control,pth,fgprint)
      self._T=self._fgprint.tree
##       self._camera={}
      self._vtktree=self.wCGNSTree(self._fgprint.tree,zlist)
      self._plot=None
      self._X=None
      self._Y=None
      self._XRangeMin=None
      self._XRangeMax=None
      self.bX.clicked.connect(self.reverseAxis)
      QObject.connect(self.sIndex1, SIGNAL("valueChanged(int)"),
                      self.setXRangeMin)
      QObject.connect(self.sIndex2, SIGNAL("valueChanged(int)"),
                      self.setXRangeMax)
      self.bSaveVTK.clicked.connect(self.b_saveVTK)

  def setXRangeMin(self,*args):
      self._XRangeMin=self.sIndex1.value()
      if ((self._XRangeMin is not None) and (self._XRangeMax is not None)): 
          self._plot.SetXRange(self._XRangeMin,self._XRangeMax)

  def setXRangeMax(self,*args):
      self._XRangeMax=self.sIndex2.value()
      if ((self._XRangeMin is not None) and (self._XRangeMax is not None)): 
          self._plot.SetXRange(self._XRangeMin,self._XRangeMax)

  def plotXY(self):
      if (self._plot is not None):
          self._vtkren.RemoveActor(self._plot)
          self._vtk.Render()
      
      value1=self._rsd[2][self.index[self._X]]
      value2=self._rsd[2][self.index[self._Y]]
      
      ar1=vtk.vtkFloatArray()
      ar1.SetNumberOfComponents(1)
      l1=value1[1].shape[0]
      for i in range(l1):
          ar1.InsertNextTuple1(value1[1][i])

      ar2=vtk.vtkFloatArray()
      ar2.SetNumberOfComponents(1)
      l2=value2[1].shape[0]
      for i in range(l2):
          ar2.InsertNextTuple1(value2[1][i])

      fieldData=vtk.vtkFieldData()
      fieldData.AllocateArrays(2)
      fieldData.AddArray(ar1)
      fieldData.AddArray(ar2)

      dataObject=vtk.vtkDataObject()
      dataObject.SetFieldData(fieldData)
      
      text=vtk.vtkTextProperty()
      text.SetFontFamilyToArial()
      text.SetFontSize(12)
      text.SetColor(0,0,0)

      textaxis=vtk.vtkTextProperty()
      textaxis.SetFontFamilyToArial()
      textaxis.SetFontSize(8)
      textaxis.SetColor(0,0,0)
      
      self._plot=vtk.vtkXYPlotActor()
      self._plot.AddDataObjectInput(dataObject)
      self._plot.SetTitle("Convergence")
      self._plot.SetTitleTextProperty(text)
      self._plot.SetXTitle(value1[0])
      self._plot.SetYTitle(value2[0][0:5])
      self._plot.SetAxisTitleTextProperty(textaxis)
      self._plot.SetAxisLabelTextProperty(textaxis)
      self._plot.SetXValuesToValue()
      self._plot.SetWidth(0.9)
      self._plot.SetHeight(0.9)
      self._plot.SetPosition(0.05, 0.05)
##       self._plot.PlotCurvePointsOn()
##       self._plot.PlotCurveLinesOff()

      self._plot.LegendOn()
      self._plot.PickableOff()

      self._plot.SetDataObjectXComponent(0, 0)
      self._plot.SetDataObjectYComponent(0, 1)
      self._plot.SetPlotColor(0,1.0,0.0,0.0)
      self._plot.SetPlotLabel(0,"Convergence")
      self._plot.GetProperty().SetColor(0.0,0.0,0.0)
      self._vtkren.AddActor2D(self._plot)
      self._vtkren.GetRenderWindow().GetInteractor().Render()
      
  def reverseAxis(self,*args):
      if ((self._X is not None) and (self._Y is not None)):
          X=self._X
          self._X=self._Y
          self._Y=X
          self.plotXY()

  def setVariableX(self,variable=[]):
      self.cVariableX.clear()
      self.cVariableX.addItem("")
      if (variable!=[]):
          for i in variable:
              self.cVariableX.addItem(i)
      QObject.connect(self.cVariableX,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getVariableX)
      
  def setVariableY(self,variable=[]):
      self.cVariableY.clear()
      self.cVariableY.addItem("")
      if (variable!=[]):
          for i in variable:
              self.cVariableY.addItem(i)
      QObject.connect(self.cVariableY,
                      SIGNAL("currentIndexChanged(int)"),
                      self.getVariableY)

  def getVariableX(self,*args):
      self._X=self.cVariableX.currentText()
      if ((self._X is not None) and (self._Y is not None)):
          self.plotXY()

  def getVariableY(self,*args):
      self._Y=self.cVariableY.currentText()
      if ((self._X is not None) and (self._Y is not None)):
          self.plotXY()      
          
  def screenshot(self):
    sshot=QPixmap.grabWindow(self.display.winId())
    sshot.save('/tmp/foo.png','png')
        
#  def leave(self):
#    self._wtop.destroy()
#  
#  def onexit(self):  
#    self._control._control.delTreeView(self._viewid,
#                                       self._fgprint.filedir,
#                                       self._fgprint.filename)
#    self.leave()     
            
  def wCGNSTree(self,T,zlist):
      o=vtk.vtkObject()
      o.SetGlobalWarningDisplay(0)
      del o
      
      self._vtk=self.display
      self._vtk.setParent(self)
      self._vtkren=vtk.vtkRenderer()      
      self._vtkwin=self._vtk.GetRenderWindow()
      self._vtkwin.SetNumberOfLayers(1)
      self._vtk.GetRenderWindow().AddRenderer(self._vtkren)
      
      self._parser=Mesh(T,zlist)
      self._rsd=self._parser.getResidus()
      if (self._rsd is not None):
          variables=[]
          self.index={}
          k=0
          for i in self._rsd[2]:
              variables+=[i[0]]
              self.index[i[0]]=k
              k+=1

          self.setVariableX(variables)
          self.setVariableY(variables)
          
      self._vtkren.SetBackground(1,1,1)
      self._vtkren.GetActiveCamera().ParallelProjectionOn()
      self._vtkren.ResetCamera()
      
      return self._vtk.GetRenderWindow()
           
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
          ## self._iren.Render()
        
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
      actors=self._vtkren.GetActors()
      actors.InitTraversal()
      actor=actors.GetNextItem()
      while actor:
          w.SetInput(actor.GetMapper().GetInput())
          actor=actors.GetNextItem()
      w.Write()   
                           
     
  def resetCam(self,pos):
      self._vtkren.ResetCamera()
      ## self._iren.Render()
        
  def close(self):
      self._vtk.GetRenderWindow().Finalize()
      QWidget.close(self)



                 


