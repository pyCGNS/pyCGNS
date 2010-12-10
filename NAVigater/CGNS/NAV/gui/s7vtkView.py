#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import Tkinter
import math, os, sys
import vtk
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
import numpy as NPY
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils    as CGU

from vtk.util.colors import *

import s7windoz

#----------------------------------------------------------------------------
class Mesh:
  def __init__(self,gcoordinates):
    self.__data=gcoordinates
    self.dx=self.__data[0][1]
    self.dy=self.__data[1][1]
    self.dz=self.__data[2][1]
    self.imax=self.dx.shape[0]
    self.jmax=self.dx.shape[1]
    self.kmax=self.dx.shape[2]
    self.color=(130,130,130)
  def setColor(self,color):
    self.color=color
  def do_vtk(self):
    pts=vtk.vtkPoints()
    pts.SetNumberOfPoints(self.imax*self.jmax*self.kmax)
    for i in range(len(self.dx.flat)):
      pts.InsertPoint(i,self.dz.T.flat[i],self.dy.T.flat[i],self.dx.T.flat[i])
    g=vtk.vtkStructuredGrid()
    g.SetPoints(pts)
    g.SetExtent(0,self.kmax-1,0,self.jmax-1,0,self.imax-1)
    d=vtk.vtkDataSetMapper()
    d.SetInput(g)
    a=vtk.vtkActor()
    a.SetMapper(d)
    a.GetProperty().SetRepresentationToWireframe()
    a.GetProperty().SetColor(*self.color)
    return a

#----------------------------------------------------------------------------
class wVTKView(s7windoz.wWindoz):
  def __init__(self,wcontrol,treefingerprint):

    self._control=wcontrol
    s7windoz.wWindoz.__init__(self,wcontrol,'CGNS.NAV: VTK view')

    self.CGNScolors=(black,red,blue,orange,green,yellow,purple)*256
    self._vtktree=self.wCGNSTree(treefingerprint.tree)

  def leave(self):
    self._wtop.destroy()
    self._control._hasvtkWindow=None
  
  def onexit(self):
    self.leave()

  def wCGNSTree(self,T):

      renWin = vtk.vtkRenderWindow()
      wren = vtk.vtkRenderer()
      renWin.AddRenderer(wren)

      for a in self.getActorList(T):
        wren.AddActor(a)

      wren.SetBackground(1,1,1)
      wren.ResetCamera()
      wren.GetActiveCamera().Elevation(60.0)
      wren.GetActiveCamera().Azimuth(30.0)
      wren.GetActiveCamera().Zoom(1.0)

      iwin = vtkTkRenderWindowInteractor(self._wtop,width=400,height=400,
                                         rw=renWin)
      iwin.pack(expand="t", fill="both")

      iren = renWin.GetInteractor()
      istyle = vtk.vtkInteractorStyleSwitch()
      iren.SetInteractorStyle(istyle)
      istyle.SetCurrentStyleToTrackballCamera()
#      iren.AddObserver("MouseMoveEvent", self.MotionCallback)
      iren.AddObserver("KeyPressEvent", self.CharCallback)
      iren.Initialize()
      renWin.Render()
      iren.Start()

      self.iren=iren
      self.wren=wren
      self.rwin=renWin
      self.isty=istyle
      self.iwin=iwin

      self._bindings={ ' ':self.b_recenter,
                       'x':self.b_xaxis,
                       'w':self.b_wire }
      return self.rwin

  def b_recenter(self,pos):
    print 'RECENTER ',pos
    self.wren.GetActiveCamera().Elevation(60.0)
    self.wren.GetActiveCamera().Azimuth(30.0)
    self.wren.GetActiveCamera().Zoom(1.0)
    
      
  def b_xaxis(self,pos):
    print 'AXIS X ',pos
      
  def b_surface(self,pos):
    print 'SURF ',pos
    actors = self.wren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToSurface()
        actor = actors.GetNextItem()
    renWin.Render()
      
  def b_wire(self,pos):
    print 'WIRE ',pos
    actors = self.wren.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        actor.GetProperty().SetRepresentationToWireframe()
        actor = actors.GetNextItem()
    self.wren.Render() 
      
  def getActorList(self,T):
    l=[]
    ml=self.getMeshes(T)
    for m in ml:
      l.append(m.do_vtk())
    return l

  def getMeshes(self,T):
    m=[]
    ml=[]
    ncolors=0
    if (T[0]==None):
     T[0]=CGK.CGNSTree_s
     T[3]=CGK.CGNSTree_ts   
    for z in CGU.getAllNodesByTypeList([CGK.Zone_ts],T):
      zT=CGU.nodeByPath(z,T)
      ncolors+=1
      for g in CGU.getAllNodesByTypeList([CGK.GridCoordinates_ts],zT):
        gT=CGU.nodeByPath(g,zT)
        cx=CGU.nodeByPath("%s/CoordinateX"%gT[0],gT)
        cy=CGU.nodeByPath("%s/CoordinateY"%gT[0],gT)
        cz=CGU.nodeByPath("%s/CoordinateZ"%gT[0],gT)
        m+=[[cx,cy,cz]]
    for ix in range(len(m)):
      mc=Mesh(m[ix])
      mc.setColor(self.CGNScolors[ix])
      ml+=[mc]
    return ml

  def MotionCallback(self,obj, event):
      pos = self.iren.GetEventPosition()
      print 'MOTION AT ',pos
      return

  def CharCallback(self,obj, event):
      iren = self.rwin.GetInteractor()
      keycode = iren.GetKeyCode()
      print '[%s]'%keycode
      pos = self.iren.GetEventPosition()
      if (self._bindings.has_key(keycode)): self._bindings[keycode](pos)
      return 

# -----------------------------------------------------------------------------
# --- last line


