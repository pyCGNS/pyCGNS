# -----------------------------------------------------------------------------
# pyS7 - CGNS/SIDS editor
# ONERA/DSNA - marc.poinot@onera.fr
# pyS7 - $Rev: 70 $ $Date: 2009-01-30 11:49:10 +0100 (Fri, 30 Jan 2009) $
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source
# tree for license information.
import S7.tk.s7globals
G___=S7.tk.s7globals.s7G


import s7cgnskeywords as K
try:
  import CGNS.cgnskeywords as CK
except ImportError:
  pass
#  s7utils.importCGNSWarning('SIDS pattern')

# --------------------------------------------------------------------
# Enumerates
def ZoneType_t_enum():
  return CK.ZoneType_l

def SimulationType_t_enum():
  return CK.SimulationType_l

def RigidGridMotionType_t_enum():
  return CK.RigidGridMotionType_l

def TurbulenceModelType_t_enum():
  return CK.TurbulenceModelType_l

def ViscosityModelType_t_enum():
  return CK.ViscosityModelType_l

def TurbulenceClosureType_t_enum():
  return CK.TurbulenceClosureType_l

def GasModelType_t_enum():
  return CK.GasModelType_l

def ThermalRelaxationModelType_t_enum():
  return CK.ThermalRelaxationModelType_l

def ChemicalKineticsModelType_t_enum():
  return CK.ChemicalKineticsModelType_l

def EMElectricFieldModelType_t_enum():
  return CK.EMElectricFieldModelType_l

def EMMagneticFieldModelType_t_enum():
  return CK.EMMagneticFieldModelType_l

def EMConductivityModelType_t_enum():
  return CK.EMConductivityModelType_l

def AverageInterfaceType_t_enum():
  return CK.AverageInterfaceType_l

def GoverningEquationsType_t_enum():
  return CK.GoverningEquationsType_l

def ElementType_t_enum():
  return CK.ElementType_l

def ArbitraryGridMotionType_t_enum():
  return CK.ArbitraryGridMotionType_l

# --- special cases
def BCTypeSimple_t_enum():
  return CK.BCTypeSimple_l

def BCTypeCompound_t_enum():
  return CK.BCTypeCompound_l

def MassUnits_t_enum():
  return CK.MassUnits_l

def LengthUnits_t_enum():
  return CK.LengthUnits_l

def TimeUnits_t_enum():
  return CK.TimeUnits_l

def TemperatureUnits_t_enum():
  return CK.TemperatureUnits_l

def AngleUnits_t_enum():
  return CK.AngleUnits_l

def LuminousIntensityUnits_t_enum():
  return CK.LuminousIntensityUnits_l

def DataClass_t_enum():
  return CK.DataClass_l

def GridLocation_t_enum():
  return CK.GridLocation_l

def GridConnectivityType_t_enum():
  return CK.GridConnectivityType_l

# --------------------------------------------------------------------
def Zone_t(pth,node,parent,tree,check):
  return 1

# --------------------------------------------------------------------
def ZoneType_t(pth,node,parent,tree,check):
  return 1
  if (parent[3] == CK.Zone_ts): return G___.SIDSmandatory
  else:                         return G___.SIDSoptional

# --------------------------------------------------------------------
def IndexRange_t(pth,node,parent,tree,check):
  if not ((node[0]==CK.PointRange_s) or (node[0]==CK.PointRangeDonor_s)):
    return 1
  if (node[2]):
    return 0
  if not ((len(node[1])==2) and (len(node[1][0]==3)) and (len(node[1][1]==3))):
    return 0
  return 1

# --------------------------------------------------------------------
def IndexRangeT2_t(pth,node,parent,tree,check):
  if not (node[0]==CK.Transform_s):
    return 1
  if (node[2]):
    return 0
  if not (len(node[1])==3):
    return 0
  for n in node[1]:
    if (n not in [1,2,3,-1,-2,-3]):
      return 0
  return 1
