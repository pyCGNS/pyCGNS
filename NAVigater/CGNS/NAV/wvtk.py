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

import vtk

# -----------------------------------------------------------------
class Q7VTK(Q7Window,Ui_Q7VTKWindow):
    def __init__(self,control,node,fgprint):
        Q7Window.__init__(self,Q7Window.VIEW_VTK,control,node,fgprint)
        widget = self.display
        widget.Initialize()
        widget.Start()
        # if you dont want the 'q' key to exit comment this.
        widget.AddObserver("ExitEvent", lambda o, e, a=self: self.quit())

        ren = vtk.vtkRenderer()
        widget.GetRenderWindow().AddRenderer(ren)

        cone = vtk.vtkConeSource()
        cone.SetResolution(8)

        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInput(cone.GetOutput())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)

        ren.AddActor(coneActor)

        # show the widget
        widget.show()

