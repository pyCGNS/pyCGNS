#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK
import CGNS.VAL.parse.messages as CGM
import CGNS.VAL.parse.generic
import CGNS.APP.sids.utils     as CGS

import string

UKZONETYPE='S004'
UKSIMTYPE='S005'
INCONSCELLPHYSDIM='S006'
UKGRIDLOCTYPE='S007'
UKGRIDCONNTYPE='S008'
BADTRANSFORM='S200'
BADVALUESHAPE='S101'
BADCELLDIM='S102'
BADPHYSDIM='S103'
BADFAMREFERENCE='S020'
BADADDFAMREFERENCE='S021'
DEFGRIDLOCATION='S030'
DEFGRIDCONNTYPE='S040'

class SIDSbase(CGNS.VAL.parse.generic.GenericParser):
  __messages={
INCONSCELLPHYSDIM:'Inconsistent PhysicalDimension/CellDimension',
UKZONETYPE:'Unknown ZoneType value',
UKGRIDLOCTYPE:'Unknown GridLocation value',
UKGRIDCONNTYPE:'Unknown GridConnectivityType value',
UKSIMTYPE:'Unknown SimulationType value',
BADVALUESHAPE:'Bad node value shape',
BADCELLDIM:'Bad value for CellDimension',
BADPHYSDIM:'Bad value for PhysicalDimension',
BADTRANSFORM:'Bad Transform values',
BADFAMREFERENCE:'Reference to unknown family [%s]',
BADADDFAMREFERENCE:'Reference to unknown additional family [%s]',
DEFGRIDLOCATION:'Default GridLocation is set to Vertex',
DEFGRIDCONNTYPE:'Default GridConnectivityType is set to Overset',
}
  # --------------------------------------------------------------------
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(SIDSbase.__messages)
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
              rs=log.push(pth,CGM.CHECK_FAIL,diagmessage,famtarget)
            else:
              self.context[famtargetpath]=True
    return rs
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,CGM.CHECK_FAIL,BADVALUESHAPE)
    else:
      cd=node[1][0]
      pd=node[1][1]
      self.context[CGK.CellDimension_s]=0
      self.context[CGK.PhysicalDimension_s]=0
      if (cd not in [1,2,3]):
        rs=log.push(pth,CGM.CHECK_FAIL,BADCELLDIM)
        self.context[CGK.CellDimension_s]=cd
      if (pd not in [1,2,3]):
        rs=log.push(pth,CGM.CHECK_FAIL,BADPHYSDIM)
        self.context[CGK.PhysicalDimension_s]=pd
      if (pd<cd):
        rs=log.push(pth,CGM.CHECK_FAIL,INCONSCELLPHYSDIM)
    return rs
  # --------------------------------------------------------------------
  def GridConnectivity_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.hasChildNodeOfType(node,CGK.GridConnectivityType_ts)):
      rs=log.push(pth,CGM.CHECK_WARN,DEFGRIDCONNTYPE)
    if (not CGU.hasChildNodeOfType(node,CGK.GridLocation_ts)):
      rs=log.push(pth,CGM.CHECK_WARN,DEFGRIDLOCATION)
    zonedonor=node[1].tostring()
    return rs
  # --------------------------------------------------------------------
  def GridConnectivityType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridConnectivityType_l)):
      rs=log.push(pth,CGM.CHECK_FAIL,UKGRIDCONNTYPE)
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridLocation_l)):
      rs=log.push(pth,CGM.CHECK_FAIL,UKGRIDLOCTYPE)
    return rs
  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.ZoneType_l)):
      rs=log.push(pth,CGM.CHECK_FAIL,UKZONETYPE)
    return rs
  # --------------------------------------------------------------------
  def SimulationType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.SimulationType_l)):
      rs=log.push(pth,CGM.CHECK_FAIL,UKSIMTYPE)
    return rs
  # --------------------------------------------------------------------
  def IntIndexDimension_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (node[0]==CGK.Transform_s):
      tr=list(node[0].flat)
      if (not CGS.transformCheckValues(tr,self.context[CGK.CellDimension_s])):
        rs=log.push(pth,CGM.CHECK_FAIL,BADTRANSFORM)
    return rs

# -----
