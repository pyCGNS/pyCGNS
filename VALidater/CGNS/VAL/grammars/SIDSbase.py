#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.PAT.cgnsutils    as CU
import CGNS.PAT.cgnstypes    as CT
import CGNS.PAT.cgnskeywords as CK

import string

class SIDSbase:
  def __init__(self,shiftstring):
    self.shiftstring=shiftstring
    self.keywordlist=CK.cgnsnames
    self.checkready=1
  # --------------------------------------------------
  def shift(self,path):
    n=string.split(path,'/')
    return len(n)*self.shiftstring
  # --------------------------------------------------------------------
  def checkLeaf(self,pth,node,parent,tree,check=1,log=None):
    if (log == None):      return 0
    if (not self.checkready): return 0
    clevel=0
    clevel=max(self.checkName(pth,node,parent,log),clevel)
    clevel=max(self.checkDataAndType(pth,node,parent,log),clevel)
    try:
      ntype=node[3]
      return apply(self.__dict__[ntype],[self,pth,node,parent,tree,check,log])
    except KeyError:
      return clevel
  # --------------------------------------------------------------------
  def checkTree(self,pth,node,parent,tree,check=1,log=None):
    log.push("\n%s\n"%(pth),'#INFO')
    clevel=self.checkLeaf(pth,node,parent,tree,check,log)
    r=[[pth,clevel]]
    parent=node
    for n in node[2]:
      r+=self.checkTree('%s/%s'%(pth,n[0]),n,parent,tree,check,log)
    return r
  # --------------------------------------------------------------------
  def getEnumerateList(self,node):
    try:
      ntype=node[3]+'_enum'
      if (ntype in self.methods):
        return getattr(self,ntype)()
    except KeyError:
      return None
  # --------------------------------------------------------------------
  def isMandatory(self,pth,node,parent,tree):
    try:
      if (node[3]==''): return 0 # link
      if (CT.types[node[3]].cardinality in [CT.C_11,CT.C_1N]): return 1
      return 0
    except TypeError: print node[0],node[1],node[3]
    
  # --------------------------------------------------------------------
  def getStatusForThisNode(self,pth,node,parent,tree):
    stat=self.isMandatory(pth,node,parent,tree)
    lpth=pth.split('/')
    if (lpth[0]==''):
      absolute=1
      if (len(lpth)>1): lpth=lpth[1:]
      else:             lpth=[]
    else:
      absolute=0
    if (node[0] in self.keywordlist): return (1,stat,absolute)
    return (0,stat,absolute)

  # --------------------------------------------------------------------
  def checkDataAndType(self,path,node,parent,log):
    clevel=0
    dt=CU.getNodeType(node)
    xt=CU.getNodeAllowedDataTypes(node)
    if (dt not in xt):
      tag='#FAIL'
      clevel=2
      msg="%sBad data type [%s] for [%s] (expected one of %s)\n"%\
           (self.shift(path),dt,node[0],xt)
      log.push(msg,tag)
    pt=CU.getNodeAllowedChildrenTypes(parent,node)
    if (pt==[]):
      pt=['CGNSBase_t','CGNSLibraryVersion_t']
    if (node[3] not in pt):
      tag='#FAIL'
      clevel=max(clevel,2)
      msg="%sBad node type [%s] for [%s]\n"%\
           (self.shift(path),node[3],node[0])
      log.push(msg,tag)
  #  else:
  #    tag=None
  #    clevel=max(clevel,0)
  #    msg="%sData type [%s] ok for [%s]\n"%(shift(path),node[3],node[0])
      
  # --------------------------------------------------------------------
  def checkName(self,path,node,parent,log):
    r=[]
    nm=node[0]
    tag=None
    clevel=0
    shft=self.shift(path)
    if (nm==''):
      tag='#FAIL'
      clevel=2
      log.push("%sName is empty string !\n"%shft,tag)
    for c in ['/','\\']:
      if (c in nm):
        tag='#FAIL'
        clevel=2
        log.push("%sForbidden char '%s' in name\n"%(shft,c),tag)
    for c in ['.','>','<','`',"'",'"',' ']:
      if (c in nm):
        tag='#WARNING'
        clevel=1
        log.push("%sPotential char '%s' issue in name\n"%(shft,c),tag)
    if (len(nm) > 32):
      tag='#WARNING'
      clevel=1
      log.push("%sName length %d is above expected 32 chars\n"%\
               (shft,len(nm)),tag)
    if (len(nm) > 32):
        tag='#WARNING'
        clevel=1
        log.push("%sName length %d is above expected 32 chars\n"%\
                 (shft,len(nm)),tag)
    if ((len(nm)>1) and ((nm[0] == ' ') or (nm[-1] == ' '))):
      tag='#WARNING'
      clevel=1
      log.push("%sName has heading/trailing space chars\n"%shft,tag)
      
    cnlist=CU.childNames(parent)
    if (cnlist):
      cnlist.remove(node[0])
      if (node[0] in cnlist):
        tag='#FAIL'
        clevel=2
        log.push("%sDuplicated node name [%s]\n"%(shft,node[0]),tag)

    return clevel
  

class SIDSpython(SIDSbase):
  def __init__(self,shiftstring=' '):
    SIDSbase.__init__(self,shiftstring)
    self.methods=[]
    for m in dir(self):
      if (m[-5:]=='_enum'): self.methods+=[m]
      
  # --------------------------------------------------------------------
  # Enumerates
  def ZoneType_t_enum(self):
    return CK.ZoneType_l

  def SimulationType_t_enum(self):
    return CK.SimulationType_l

  def RigidGridMotionType_t_enum(self):
    return CK.RigidGridMotionType_l

  def TurbulenceModelType_t_enum(self):
    return CK.TurbulenceModelType_l

  def ViscosityModelType_t_enum(self):
    return CK.ViscosityModelType_l

  def TurbulenceClosureType_t_enum(self):
    return CK.TurbulenceClosureType_l

  def GasModelType_t_enum(self):
    return CK.GasModelType_l

  def ThermalRelaxationModelType_t_enum(self):
    return CK.ThermalRelaxationModelType_l

  def ChemicalKineticsModelType_t_enum(self):
    return CK.ChemicalKineticsModelType_l

  def EMElectricFieldModelType_t_enum(self):
    return CK.EMElectricFieldModelType_l

  def EMMagneticFieldModelType_t_enum(self):
    return CK.EMMagneticFieldModelType_l

  def EMConductivityModelType_t_enum(self):
    return CK.EMConductivityModelType_l

  def AverageInterfaceType_t_enum(self):
    return CK.AverageInterfaceType_l

  def GoverningEquationsType_t_enum(self):
    return CK.GoverningEquationsType_l

  def ElementType_t_enum(self):
    return CK.ElementType_l

  def ArbitraryGridMotionType_t_enum(self):
    return CK.ArbitraryGridMotionType_l

  # --- special cases
  def BCTypeSimple_t_enum(self):
    return CK.BCTypeSimple_l

  def BCTypeCompound_t_enum(self):
    return CK.BCTypeCompound_l

  def MassUnits_t_enum(self):
    return CK.MassUnits_l

  def LengthUnits_t_enum(self):
    return CK.LengthUnits_l

  def TimeUnits_t_enum(self):
    return CK.TimeUnits_l

  def TemperatureUnits_t_enum(self):
    return CK.TemperatureUnits_l

  def AngleUnits_t_enum(self):
    return CK.AngleUnits_l

  def LuminousIntensityUnits_t_enum(self):
    return CK.LuminousIntensityUnits_l

  def DataClass_t_enum(self):
    return CK.DataClass_l

  def GridLocation_t_enum(self):
    return CK.GridLocation_l

  def GridConnectivityType_t_enum(self):
    return CK.GridConnectivityType_l

  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,check,log):
    rs=1
    r=0
    msg='## No GridCoordinates in this Zone_t'
    for cn in CU.childNames(node):
      if (cn!='GridCoordinates'):
        r=1
        break
    if (not r): log.push("\n%s\n"%(msg),'#FAIL')
    rs*=r
    r=0
    msg='## No FlowSolution# found for output definition'
    for cn in CU.childNames(node):
      if (    (len(cn)>12)
          and (cn[:13]=='FlowSolution#')
          and (cn!='FlowSolution#Init')):
        r=1
        break
    if (not r): log.push("\n%s\n"%(msg),'#WARNING')
    rs*=r
    r=0
    msg='## No FlowSolution#Init found for field initialization'
    for cn in CU.childNames(node):
      if (cn=='FlowSolution#Init'):
        r=1
        break
    if (not r): log.push("\n%s\n"%(msg),'#WARNING')
    rs*=r
    return rs

  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,check,log):
    rs=1
    r=0
    msg='## No Zone_t found in this CGNSBase_t'
    r=CU.hasChildNodeOfType(node,CK.Zone_ts)
    if (not r): log.push("\n%s\n"%(msg),'#FAIL')
    rs*=r
    r=0
    msg='## No ReferenceState found in this CGNSBase_t'
    for cn in CU.childNames(node):
      if (cn!=CK.ReferenceState_s):
        r=1
        break
    if (not r): log.push("\n%s\n"%(msg),'#WARNING')
    rs*=r
    return rs

  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,check,log):
    msg='## Only Structured ZoneType_t is allowed'
    r=CU.stringValueMatches(node,'Structured')
    if (not r): log.push("\n%s\n"%(msg),'#FAIL')
    return 1

  # --------------------------------------------------------------------
  def IndexRange_t(self,pth,node,parent,tree,check,log):
    if not ((node[0]==CK.PointRange_s) or (node[0]==CK.PointRangeDonor_s)):
      return 1
    if (node[2]):
      return 0
    if not ((len(node[1])==2) and (len(node[1][0]==3)) and (len(node[1][1]==3))):
      return 0
    return 1

  # --------------------------------------------------------------------
  def IndexRangeT2_t(self,pth,node,parent,tree,check,log):
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

# -----
