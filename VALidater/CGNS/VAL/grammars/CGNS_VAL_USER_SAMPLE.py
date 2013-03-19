#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils          as CGU
import CGNS.PAT.cgnstypes          as CGT
import CGNS.PAT.cgnskeywords       as CGK
import CGNS.VAL.parse.messages     as CGM
import CGNS.VAL.parse.generic

messagetable=(
('U101',CGM.CHECK_WARN,'No Zone in this Base'),
('U102',CGM.CHECK_WARN,'No GridCoordinates in this Zone'),
('U105',CGM.CHECK_FAIL,'At least one structured Zone is required in the Base'),
('U103',CGM.CHECK_WARN,'No ReferenceState found at Base level'),
('U104',CGM.CHECK_WARN,'No ReferenceState found at Zone level'),
('U107',CGM.CHECK_WARN,'No FlowSolution# found for output definition'),
('U108',CGM.CHECK_WARN,'No FlowSolution#Init found for fields initialisation'),
('U106',CGM.CHECK_FAIL,'Transform is not right-handed (direct)'),
('U109',CGM.CHECK_FAIL,'Cannot handle such GridLocation [%s]'),
('U110',CGM.CHECK_FAIL,'Cannot handle such ElementType [%s]'),
)

USER_MESSAGES={}
for (k,l,m) in messagetable:
  USER_MESSAGES[k]=(l,m)

# -----------------------------------------------------------------------------
class CGNS_VAL_USER_Checks(CGNS.VAL.parse.generic.GenericParser):
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(USER_MESSAGES)
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    zt=CGU.hasChildName(node,CGK.ZoneType_s)
    if (zt is not None):
      if (CGU.stringValueMatches(zt,CGK.Structured_s)):    
        shp=(parent[1][0],3)
        idxdim=3
      else:
        shp=(1,)
        idxdim=1
      if (CGU.getShape(node)!=shp):
        rs=log.push(pth,'S009',CGU.getShape(node))
      elif (CGU.stringValueMatches(zt,CGK.Structured_s)):
        zd=node[1]
        for nd in range(idxdim):
          if ((zd[nd][1]!=zd[nd][0]-1) or (zd[nd][2]!=0)):
            rs=log.push(pth,'S010')
    return rs
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    (cd,pd)=(0,0)
    if (not CGU.hasChildNodeOfType(node,CGK.Zone_ts)):
      rs=log.push(pth,'U101')
    else:
      target=[CGK.CGNSTree_ts,node[0],CGK.Zone_ts,CGK.ZoneType_s]
      plist=CGU.getAllNodesByTypeOrNameList(tree,target)
      found=False
      for p in plist:
        if (CGU.stringValueMatches(CGU.getNodeByPath(tree,p),
                                   CGK.Structured_s)):
          found=True
      if (not found):
        rs=log.push(pth,'U102')
    if (not CGU.hasChildNodeOfType(node,CGK.ReferenceState_ts)):
      rs=log.push(pth,'U103')
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,'S009',CGU.getShape(node))
    else:
      cd=node[1][0]
      pd=node[1][1]
      allowedvalues=((1,1),(1,2),(1,3),(2,2),(2,3),(3,3))
      if ((cd,pd) not in allowedvalues):
        rs=log.push(pth,'S010',(cd,pd))
    self.context[CGK.CellDimension_s]=cd
    self.context[CGK.PhysicalDimension_s]=pd
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    val=node[1].tostring()
    if (val not in [CGK.Vertex_s,CGK.CellCenter_s,CGK.FaceCenter_s]):
        rs=log.push(pth,'U109',val)
    return rs
# -----
