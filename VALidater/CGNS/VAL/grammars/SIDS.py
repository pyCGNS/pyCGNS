#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK
import CGNS.APP.sids.utils     as CGS
import CGNS.VAL.parse.messages as CGM
import CGNS.VAL.parse.generic

messagetable=(
('UKZONETYPE'         ,'S101',CGM.CHECK_FAIL,'Unknown ZoneType value'),
('UKSIMTYPE'          ,'S102',CGM.CHECK_FAIL,'Unknown SimulationType value'),
('UKGRIDLOCTYPE'      ,'S103',CGM.CHECK_FAIL,'Unknown GridLocation value'),
('UKGRIDCONNTYPE'     ,'S104',CGM.CHECK_FAIL,'Unknown GridConnectivityType value'),
('UKDATACLASS'        ,'S105',CGM.CHECK_FAIL,'Unknown DataClass value'),
('UKBCDATATYPE'       ,'S106',CGM.CHECK_FAIL,'Unknown BCDataType value'),
('UKRIGIDMOTYPE'      ,'S107',CGM.CHECK_FAIL,'Unknown RigidMotionType value'),
('UKBCTYPE'           ,'S108',CGM.CHECK_FAIL,'Unknown BCType value'),
('UKELEMTYPE'         ,'S109',CGM.CHECK_FAIL,'Unknown ElementType value'),
('UKMASSUNIT'         ,'S110',CGM.CHECK_FAIL,'Unknown MassUnit value'),
('UKTIMEUNIT'         ,'S111',CGM.CHECK_FAIL,'Unknown TimeUnit value'),
('UKLENGTHUNIT'       ,'S112',CGM.CHECK_FAIL,'Unknown LengthUnit value'),
('UKTEMPUNIT'         ,'S113',CGM.CHECK_FAIL,'Unknown TemperatureUnit value'),
('UKANGLEUNIT'        ,'S114',CGM.CHECK_FAIL,'Unknown AngleUnit value'),
('UKECUUNIT'          ,'S115',CGM.CHECK_FAIL,'Unknown ElectricCurrentUnit value'),
('UKSUBSTUNIT'        ,'S116',CGM.CHECK_FAIL,'Unknown SubstanceAmountUnit value'),
('UKLUMUNIT'          ,'S117',CGM.CHECK_FAIL,'Unknown LuminousIntensityUnit value'),

('BADVALUESHAPE'      ,'S191',CGM.CHECK_FAIL,'Bad node value shape'),

('DEFGRIDLOCATION'    ,'S151',CGM.CHECK_WARN,'Default GridLocation is set to Vertex'),
('DEFGRIDCONNTYPE'    ,'S152',CGM.CHECK_WARN,'Default GridConnectivityType is set to Overset'),

('INCONSCELLPHYSDIM'  ,'S201',CGM.CHECK_FAIL,'Inconsistent PhysicalDimension/CellDimension'),
('BADCELLDIM'         ,'S202',CGM.CHECK_FAIL,'Bad value for CellDimension'),
('BADPHYSDIM'         ,'S203',CGM.CHECK_FAIL,'Bad value for PhysicalDimension'),
('BADTRANSFORM'       ,'S204',CGM.CHECK_FAIL,'Bad Transform values'),
('BADELEMSZBND'       ,'S205',CGM.CHECK_FAIL,'Bad ElementSizeBoundary value'),

('BADFAMREFERENCE'    ,'S301',CGM.CHECK_FAIL,'Reference to unknown family [%s]'),
('BADADDFAMREFERENCE' ,'S302',CGM.CHECK_FAIL,'Reference to unknown additional family [%s]'),
)

SIDS_MESSAGES={}
for (v,k,l,m) in messagetable:
  locals()[v]=k
  SIDS_MESSAGES[k]=(l,m)

# ------------------------------------------------------------------------------
class SIDSbase(CGNS.VAL.parse.generic.GenericParser):
  # --------------------------------------------------------------------
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(SIDS_MESSAGES)
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,BADVALUESHAPE)
    else:
      cd=node[1][0]
      pd=node[1][1]
      self.context[CGK.CellDimension_s]=cd
      self.context[CGK.PhysicalDimension_s]=pd
      if (cd not in [1,2,3]):
        rs=log.push(pth,BADCELLDIM)
      if (pd not in [1,2,3]):
        rs=log.push(pth,BADPHYSDIM)
      if (pd<cd):
        rs=log.push(pth,INCONSCELLPHYSDIM)
    return rs
  # --------------------------------------------------------------------
  def SimulationType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.SimulationType_l)):
      rs=log.push(pth,UKSIMTYPE)
    return rs
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.hasChildNodeOfType(node,CGK.FamilyName_ts)):
      basepath=[CGK.CGNSTree_ts,parent[0],node[0]]
      searchpath=basepath+[CGK.FamilyName_ts]
      famlist1=CGU.getAllNodesByTypeOrNameList(tree,searchpath)
      searchpath=basepath+[CGK.AdditionalFamilyName_ts]
      famlist2=CGU.getAllNodesByTypeOrNameList(tree,searchpath)
      for (famlist, diagmessage) in ((famlist1,BADFAMREFERENCE),
                                     (famlist2,BADADDFAMREFERENCE)):
        for fampath in famlist:
          famtarget=CGU.getNodeByPath(tree,fampath)[1].tostring()
          famtargetpath="/%s/%s"%(parent[0],famtarget)
          if (not self.context.has_key(famtargetpath)):
            famtargetnode=CGU.getNodeByPath(tree,famtargetpath)
            if (famtargetnode is None):
              rs=log.push(pth,diagmessage,famtarget)
            else:
              self.context[famtargetpath]=True
    return rs
  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.ZoneType_l)):
      rs=log.push(pth,UKZONETYPE)
    return rs
  # --------------------------------------------------------------------
  def GridConnectivity_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.hasChildNodeOfType(node,CGK.GridConnectivityType_ts)):
      rs=log.push(pth,DEFGRIDCONNTYPE)
    if (not CGU.hasChildNodeOfType(node,CGK.GridLocation_ts)):
      rs=log.push(pth,DEFGRIDLOCATION)
    zonedonor=node[1].tostring()
    return rs
  # --------------------------------------------------------------------
  def GridConnectivityType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridConnectivityType_l)):
      rs=log.push(pth,UKGRIDCONNTYPE)
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridLocation_l)):
      rs=log.push(pth,UKGRIDLOCTYPE)
    return rs
  # --------------------------------------------------------------------
  def IntIndexDimension_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (node[0]==CGK.Transform_s):
      tr=list(node[0].flat)
      if (not CGS.transformCheckValues(tr,self.context[CGK.CellDimension_s])):
        rs=log.push(pth,BADTRANSFORM)
    return rs
  # --------------------------------------------------------------------
  def Elements_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,BADVALUESHAPE)
    else:
      et=node[1][0]
      eb=node[1][1]
      self.context[CGK.ElementType_s]=et
      self.context[CGK.ElementSizeBoundary_s]=eb
      if (et not in range(0,len(CGK.ElementType)+1)):
        rs=log.push(pth,UKELEMTYPE)
      if (eb==0): bad_eb=False
      elif (eb<0): bad_eb=True
      else:
        bad_eb=True
        ecnode=CGU.getNodeByPath(tree,pth+'/'+CGK.ElementRange_s)
        if (    (ecnode is not None)
            and (CGU.getShape(node)==(2,))
            and (CGU.getValueDataType(ecnode)==CGK.I4)
            and (ecnode[1][1]>eb)): bad_eb=False
      if (bad_eb):
        rs=log.push(pth,BADELEMSZBND)
    return rs

# -----
