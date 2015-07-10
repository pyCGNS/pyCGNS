#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newUserDefinedData(None,'Solver#Param')
CGL.newDataArray(data,'axis_ang_1',   CGU.setIntegerAsArray(60))
CGL.newDataArray(data,'axis_ang_2',   CGU.setIntegerAsArray(1))
CGL.newDataArray(data,'axis_pnt_x',   CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'axis_pnt_y',   CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'axis_pnt_z',   CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'axis_vct_x',   CGU.setDoubleAsArray(1.0))
CGL.newDataArray(data,'axis_vct_y',   CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'axis_vct_z',   CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'interpol_tool',CGU.setStringAsArray('adt'))
CGL.newDataArray(data,'motion',       CGU.setStringAsArray('mobile'))
CGL.newDataArray(data,'omega',        CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'transl_speed', CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'transl_vct_x', CGU.setDoubleAsArray(1.0))
CGL.newDataArray(data,'transl_vct_y', CGU.setDoubleAsArray(0.0))
CGL.newDataArray(data,'transl_vct_z', CGU.setDoubleAsArray(0.0))
    
status='0.1'
comment='Motion parameter set'
pattern=[data, status, comment]
