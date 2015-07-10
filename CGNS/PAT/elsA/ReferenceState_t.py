#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnserrors   as CGE
import CGNS.PAT.cgnskeywords as CGK
import numpy                 as NPY

data=CGL.newReferenceState(b,CGK.ReferenceState_s)
d=CGL.newDescriptor(rs,CGK.ReferenceStateDescription_s,CGU.setStringAsArray("Global reference state"))
d=CGL.newDataArray(rs,CGK.Mach_s,CGU.setDoubleAsArray(0.2))
d=CGL.newDataArray(rs,'AngleofAttack',CGU.setDoubleAsArray(7.0))
d=CGL.newDataArray(rs,'BetaAngle',CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.Reynolds_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.LengthReference_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.Coef_Area_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.VelocityMagnitude_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.Density_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.MomentumX_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.MomentumY_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.MomentumZ_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.EnergyStagnationDensity_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,'TubulentSANuTilde',CGU.setDoubleAsArray(0.0))
d=CGL.newDataArray(rs,CGK.Pressure_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.Temperature_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.PressureStagnation_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.TemperatureStagnation_s,CGU.setDoubleAsArray(1.0))
d=CGL.newDataArray(rs,CGK.Coef_PressureDynamic_s,CGU.setDoubleAsArray(1.0))
    
status='0.1'
comment='ONERA/elsA CFD pattern'
pattern=[data, status, comment]
