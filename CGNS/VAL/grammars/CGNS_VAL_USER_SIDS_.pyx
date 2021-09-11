#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils      as CGU
import CGNS.PAT.cgnstypes      as CGT
import CGNS.PAT.cgnskeywords   as CGK
import CGNS.PAT.cgnslib        as CGL
import CGNS.VAL.parse.utils    as CGS
import CGNS.VAL.parse.messages as CGM
import CGNS.VAL.parse.generic

from CGNS.VAL.grammars import valutils as val_u
from CGNS.VAL.grammars.etablesids import messagetable

import string
import numpy as NPY
cimport numpy as NCY
from cpython cimport bool as py_bool 

SIDS_MESSAGES={}
for (k,l,m) in messagetable:
  SIDS_MESSAGES[k]=(l,m)

# PARSE ORDER IS VERY IMPORTANT FOR CONTEXT
# WIDTH FIRST IS HARD-CODED IN generic.py

# -----------------------------------------------------------------------------
class CGNS_VAL_USER_Checks(CGNS.VAL.parse.generic.GenericParser):
  # --------------------------------------------------------------------
  def __init__(self,log):
    CGNS.VAL.parse.generic.GenericParser.__init__(self,log)
    self.log.addMessages(SIDS_MESSAGES)
  # --------------------------------------------------------------------
  def CGNSBase_t(self,pth,node,parent,tree,log):
    self.dbg("CGNSBase_t",pth)
    rs=CGM.CHECK_OK
    self.context[CGK.DataClass_s][pth]=None
    self.context[CGK.DimensionalUnits_s][pth]=None
    self.context[CGK.NumberOfSteps_s][pth]=0
    self.context[CGK.CellDimension_s][pth]=0
    self.context[CGK.PhysicalDimension_s][pth]=0
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,'S0191')
    else:
      cd=node[1][0]
      pd=node[1][1]
      self.context[CGK.CellDimension_s][pth]=cd
      self.context[CGK.PhysicalDimension_s][pth]=pd
      if (cd not in [1,2,3]):
        rs=log.push(pth,'S0202')
      if (pd not in [1,2,3]):
        rs=log.push(pth,'S0203')
      if (pd<cd):
        rs=log.push(pth,'S0201')
    if (not CGU.hasChildType(node,CGK.Zone_ts)):
      rs=log.push(pth,'S0601')
    return rs
  # --------------------------------------------------------------------
  def SimulationType_t(self,pth,node,parent,tree,log):
    self.dbg("SimulationType_t",pth)
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.SimulationType_l)):
      rs=log.push(pth,'S0102')
    return rs
  # --------------------------------------------------------------------
  def Zone_t(self,pth,node,parent,tree,log):
    self.dbg("Zone_t",pth)
    rs=CGM.CHECK_OK
    zt=CGU.hasChildName(node,CGK.ZoneType_s) # zone type
    zv=[] # vertexsize
    zc=[] # cellsize
    isStructured=True
    if (zt is not None):
      if (CGU.stringValueMatches(zt,CGK.Unstructured_s)):
        isStructured=False
    if 1:
      if (not isStructured):
        self.context[CGK.IndexDimension_s][pth]=1
      else:
        cd=self.context[CGK.CellDimension_s][pth] # default is Structured
        self.context[CGK.IndexDimension_s][pth]=cd
      id=self.context[CGK.IndexDimension_s][pth]
      bad_value = False
      shp=(id,3)
      if (CGU.getShape(node)!=shp):
        rs=log.push(pth,'S0192',CGU.getShape(node),shp,node[0])
        bad_value = True
      elif (isStructured):
        zd=node[1]
        for nd in range(id):
          zv.append(zd[nd][0]) # add VertexSize to context
          zc.append(zd[nd][1]) # add CellSize to context
          if ((zd[nd][1]!=zd[nd][0]-1) or (zd[nd][2]!=0)):
            rs=log.push(pth,'S0010')
            bad_value = True
      elif (not isStructured):
        zd=node[1]
        nd=0
        zv.append(zd[nd][0]) # add VertexSize to context
        zc.append(zd[nd][1]) # add CellSize to context
        #if (zd[nd][0]>=zd[nd][1]) or (zd[nd][2]!=0)):
        #  rs=log.push(pth,'S0010')
        #  bad_value = True
        if (not CGU.hasChildType(node,CGK.Elements_ts)):
          rs=log.push(pth,'S0605')       
      self.context[CGK.VertexSize_s][pth]=tuple(zv)
      self.context[CGK.CellSize_s][pth]=tuple(zc)
      if (    not bad_value 
          and not isStructured
          and CGU.hasChildType(node,CGK.Elements_ts)):
        # Checking correct combination of element ranges using of a
        # table initialized at 0 and incremented by 1 for each element
        # found in ElementRange children
        cellsize = self.context[CGK.CellSize_s][pth][0] 
        tall  = NPY.zeros(cellsize,dtype='i4', order='F')
        has_nface = False # initialize
        has_ngon  = False # initialize
        facesize  = 0     # initialize
        erl = dict(zip(range(len(CGK.ElementType)),
                       NPY.empty([len(CGK.ElementType),0]).tolist()))
        erpathlist=CGU.getPathsByTypeOrNameList(node,[node[3],
                                                      CGK.Elements_ts,
                                                      CGK.ElementRange_s])
        for erpath in erpathlist: # loop on element range paths
          er  = CGU.getNodeByPath(node,erpath) # element range node
          erp = CGU.getParentFromNode(tree,er) # element_t node
          et  = erp[1][0]                      # element type (integer)
          shp = (id,2)
          if ((CGU.getShape(er) == shp)
              and (CGU.getValueDataType(er) in [CGK.I4, CGK.I8])):
            # enlarge tall table if necessary
            min_i  = len(tall)+1
            max_i  = max(er[1][0])
            if (max_i>=min_i):
              t      = NPY.zeros(max_i-min_i+1,dtype='i4', order='F')
              tall   = NPY.concatenate((tall,t)) 
            if (all(er[1][0] > 0)): 
              tall[range(min(er[1][0])-1,max(er[1][0]))] += 1
              erl[et].append([min(er[1][0]),max(er[1][0])]) 
          if   (et == CGK.NFACE_n): has_nface = True                  
          elif (et == CGK.NGON_n):  has_ngon  = True            
        if ( erpathlist and any(tall != 1)):
          rs=log.push(pth,'S0209') 
        if (has_nface and not has_ngon):
          rs=log.push(pth,'S0198',CGK.NGON_n_s,CGK.NFACE_n_s)
        self.context[CGK.ElementsSize_s][pth]=len(tall) 
        self.context[CGK.ElementRangeList_s][pth]=erl
      elif (    not bad_value 
            and isStructured):
        # Structured zone should not have Elements_t
        if (CGU.hasChildType(node,CGK.Elements_ts)): 
          rs=log.push(pth,'S0608')
        # check all boundary faces are either on a BC or a
        # GridConnectivity node Only for structured zones here, test
        # for unstructured zones is performed in Elements_t
        # get IndexRange and IndexArray on BC and GridConnectivity
        (rl,al) = val_u.getIndicesOnBCandGC(tree,pth,id)
        # create boundary face tables, full of 0, face-centered
        bndfaces = val_u.initBndFaces(zd) 
        for r in rl:
           # get face number and indices, vertex-centered
          (bnd,imin,imax,jmin,jmax)=val_u.getFaceNumber(r,zd)
          if (bnd in range(cd*2)):
            # add 1 to faces in range r         
            bndfaces[bnd][imin-1:imax-1,jmin-1:jmax-1] += 1
        for bnd in range(cd*2): # loop on boundary faces
          if (any(bndfaces[bnd].flatten() == 0)): # not defined faces
            rs=log.push(pth,'S0702',
                        val_u.bdnName(bnd),NPY.where(bndfaces[bnd] == 0)) 
          if (any(bndfaces[bnd].flatten() > 1)): # doubly defined faces
            rs=log.push(pth,'S0703',
                        val_u.bdnName(bnd),NPY.where(bndfaces[bnd] > 1))   
    if (CGU.hasChildNodeOfType(node,CGK.FamilyName_ts)):
      basepath=[CGK.CGNSTree_ts,parent[0],node[0]]
      searchpath=basepath+[CGK.FamilyName_ts]
      famlist1=CGU.getAllNodesByTypeOrNameList(tree,searchpath)
      searchpath=basepath+[CGK.AdditionalFamilyName_ts]
      famlist2=CGU.getAllNodesByTypeOrNameList(tree,searchpath)
      for (famlist, diagmessage) in ((famlist1,'S0301'),
                                     (famlist2,'S0302')):
        for fampath in famlist:
          famdefinition=CGU.getNodeByPath(tree,fampath)
          if (famdefinition[1] is None):
            rs=log.push(pth,'S0300')
          else:
            famtarget=famdefinition[1].tostring().rstrip()
            famtargetpath="/%s/%s"%(parent[0],famtarget)
            if (famtargetpath not in self.context):
              famtargetnode=CGU.getNodeByPath(tree,famtargetpath)
              if (famtargetnode is None):
                rs=log.push(pth,diagmessage,famtarget)
              else:
                self.context[famtargetpath][pth]=True
    if (not CGU.hasChildType(node,CGK.GridCoordinates_ts)):
      rs=log.push(pth,'S0602')
    elif (not CGU.hasChildName(node,CGK.GridCoordinates_s)):
      rs=log.push(pth,'S0603')      
    if (not CGU.hasChildType(node,CGK.ZoneBC_ts)):
      rs=log.push(pth,'S0604')     
    return rs
  # --------------------------------------------------------------------
  def ZoneType_t(self,pth,node,parent,tree,log):
    self.dbg("ZoneType_t",pth)
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.ZoneType_l)):
      rs=log.push(pth,'S0101')
    return rs
  # --------------------------------------------------------------------
  def CGNSTree_t(self,pth,node,parent,tree,log):
    self.dbg("CGNSTree_t",pth)
    rs=CGM.CHECK_OK
    if (not CGU.hasChildType(node,CGK.CGNSBase_ts)):
      rs=log.push(pth,'S0600')
    return rs
  # --------------------------------------------------------------------
  def ReferenceState_t(self,pth,node,parent,tree,log):
    self.dbg("ReferenceState_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def Family_t(self,pth,node,parent,tree,log):
    self.dbg("Family_t",pth)
    rs=CGM.CHECK_OK
    famname=node[0]
    basename=parent[0]
    found=False
    for fpth in CGU.getAllParentTypePaths(CGK.FamilyName_ts):
      fpth[1]=basename
      pfamlist=CGU.getAllNodesByTypeOrNameList(tree,fpth)
      for pfam in pfamlist:
        nfam=CGU.getNodeByPath(tree,pfam)
        if (    (nfam is not None)
            and (nfam[1] is not None)
            and (nfam[1].tostring().rstrip()==famname)):
          found=True
          break
      if (found): break
    if (not found):
      rs=log.push(pth,'S0303')
    return rs
  # --------------------------------------------------------------------
  def IndexArray_t(self,pth,node,parent,tree,log):
    self.dbg("IndexArray_t",pth)
    rs=CGM.CHECK_OK
    if (node[0] not in CGU.getAuthNames(node)):
      rs=log.push(pth,'S0122',node[0],node[3],CGU.getAuthNames(node)) 
    # Getting refered-to zone (current or donor)
    zpth = None
    if (parent[3] in CGU.getAuthParentTypes(node)):
      if (node[0] in [CGK.PointList_s,CGK.InwardNormalList_s]):
        zpth = pth
      elif (node[0] in [CGK.PointListDonor_s,CGK.CellListDonor_s] 
            and parent[3] in [CGK.GridConnectivity1to1_ts,
                              CGK.GridConnectivity_ts]):
        if (CGU.getValueDataType(parent) in [CGK.C1]):
          zdonorname=parent[1].tostring().rstrip()
          basename=val_u.getBase(pth,node,parent,tree,log)
          targetpath='/%s/%s'%(basename,zdonorname)
          zdonor=CGU.getNodeByPath(tree,targetpath)
          if (zdonor is not None): zpth = targetpath
    if (zpth is not None):
      ztype = val_u.getZoneType(tree,zpth)
      ntype = CGU.getValueDataType(node)
      idim  = self.context[CGK.IndexDimension_s][zpth]
      if (    (CGU.getAuthDataTypes(node) is not None) 
          and (ntype not in CGU.getAuthDataTypes(node))): # checking data type 
        rs=log.push(pth,'S0199',ntype,CGU.getAuthDataTypes(node))
      else :
        if (node[0] == CGK.InwardNormalList_s):
          if (ntype not in [CGK.R4, CGK.R8]): # checking data type
            rs=log.push(pth,'S0199',ntype,[CGK.R4, CGK.R8])
          shp = CGU.getShape(node)
          listlength = val_u.getListLength(parent)
          if (listlength > 0):
            targetshp = (self.context[CGK.PhysicalDimension_s][zpth],listlength)
            if (shp != targetshp): # checking shape
              rs=log.push(pth,'S0192',shp,targetshp,node[0])
          else:
            if ((len(shp)!=2)
                or (shp[0]!=self.context[CGK.PhysicalDimension_s][zpth])): 
              rs=log.push(pth,'S0192',shp,
                          '(%s,?)' %self.context[CGK.PhysicalDimension_s][zpth],
                          node[0])            
        else: # node[0] != CGK.InwardNormalList_s
          if (CGU.getShape(node)[0]!=idim):  # checking shape
            rs=log.push(pth,'S0192',CGU.getShape(node),idim,node[0])
          if (ntype not in [CGK.I4,CGK.I8]): # checking data type 
            rs=log.push(pth,'S0199',ntype,[CGK.I4,CGK.I8])
          else:
            pl=node[1]
            if (pl.size!=len(set(pl.flatten().tolist()))):
              rs=log.push(pth,'S0208')
            if (any(pl.flatten()<1)): rs=log.push(pth,'S0206')
            if (parent[3] == CGK.BCDataSet_ts):
              refnode=CGU.getNodeByPath(tree,CGU.getPathAncestor(pth,level=2))
            else:
              refnode=parent
            (gridloc,rs)=val_u.getGridLocation(refnode,pth=pth,log=log,rs=rs)
            # cell data required for cell list donor
            if (    gridloc != CGK.CellCenter_s 
                and node[0] == CGK.CellListDonor_s): 
              rs=log.push(pth,'S0218',gridloc,node[0],[CGK.CellCenter_s])
            elif (    (gridloc==CGK.Vertex_s
                       or
                       (    gridloc.endswith(CGK.FaceCenter_s)
                            and ztype==CGK.Structured_s
                       )
                      )
                  and (len(self.context[CGK.VertexSize_s][zpth])==idim)
                 ):
              dd=self.context[CGK.VertexSize_s][zpth]
              for d in range(idim):
                if (any(pl[d]>dd[d])):
                  rs=log.push(pth,'S0211',CGK.Vertex_s,d+1,idim,[1,dd[d]])
            elif (    (gridloc==CGK.CellCenter_s) # Cell center data 
                  and (ztype==CGK.Structured_s) # for structured zone 
                  and (len(self.context[CGK.CellSize_s][zpth])==idim)):
              dd=self.context[CGK.CellSize_s][zpth]
              for d in range(idim):
                if (any(pl[d]>dd[d])):
                  rs=log.push(pth,'S0211',
                              CGK.CellCenter_s,d+1,idim,[1,dd[d]])
            else:
              if (ztype==CGK.Unstructured_s): # Unstructured zone
                imin=1   
                imax=self.context[CGK.ElementsSize_s][zpth]
                if (any(pl[0]<imin) or any(pl[0]>imax)):
                  rs=log.push(pth,'S0211',gridloc,1,idim,[imin,imax])
                _lv=val_u.getLevelFromGridLoc(gridloc)
                _pd=self.context[CGK.PhysicalDimension_s][zpth]
                _er=self.context[CGK.ElementRangeList_s][zpth]
                erl=val_u.getElementTypeRangeList(_lv,_pd,_er)
                rs=val_u.allIndexInElementRangeList(pl[0],erl,pth,log,rs)
    return rs
  # --------------------------------------------------------------------
  def DiscreteData_t(self,pth,node,parent,tree,log):
    self.dbg("DiscreteData_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def IndexRange_t(self,pth,node,parent,tree,log):
    self.dbg("IndexRange_t",pth)
    rs=CGM.CHECK_OK
    if (node[0] not in CGU.getAuthNames(node)):
      rs=log.push(pth,'S0122',node[0],node[3],CGU.getAuthNames(node))
    # ---
    # Get parent zone (current or donor)
    zpth = None
    if (parent[3] in CGU.getAuthParentTypes(node)):
      if (node[0] in [CGK.PointRange_s,CGK.ElementRange_s]):
        zpth = pth
      elif ((node[0] in [CGK.PointRangeDonor_s]) 
            and (parent[3] in [CGK.GridConnectivity1to1_ts,
                               CGK.GridConnectivity_ts])):
        if (CGU.getValueDataType(parent) in [CGK.C1]):
          zdonorname=parent[1].tostring().rstrip()
          basename=val_u.getBase(pth,node,parent,tree,log)
          targetpath='/%s/%s'%(basename,zdonorname)
          zdonor=CGU.getNodeByPath(tree,targetpath)
          if (zdonor is not None): zpth = targetpath
      elif (    node[0] in [CGK.PointRangeDonor_s] 
            and parent[3] not in [CGK.GridConnectivity1to1_ts,
                                  CGK.GridConnectivity_ts]):
        rs=log.push(pth,'S0005',node[0],node[3])
      else:
        zpth = pth
    if (zpth is not None):          
      shp=(self.context[CGK.IndexDimension_s][zpth],2)
      ntype = CGU.getValueDataType(node)
      ztype = val_u.getZoneType(tree,zpth)
      checkData=True
      if (node[0]==CGK.ElementRange_s):
        eshp=(2,)
        checkData=False
        if (CGU.getShape(node)!=eshp):
          rs=log.push(pth,'S0192',CGU.getShape(node),eshp,node[0])
      elif (CGU.getShape(node)!=shp):
        rs=log.push(pth,'S0192',CGU.getShape(node),shp,node[0])
        checkData=False
      if (ntype not in CGU.getAuthDataTypes(node)):
        rs=log.push(pth,'S0199',ntype,CGU.getAuthDataTypes(node))
      if checkData:
        pr=node[1]
        if (any(pr.flatten()<1)): rs=log.push(pth,'S0206')
        # checking out of range (<1) 
        for d in range(self.context[CGK.IndexDimension_s][zpth]):
          if (pr[d][0]>pr[d][1]): rs=log.push(pth,'S0207')
          # checking not ordered element range         
        if (node[0] in [CGK.PointRange_s,CGK.PointRangeDonor_s]):
          if (parent[3] == CGK.BCDataSet_ts):
            refnode = CGU.getNodeByPath(tree,CGU.getPathAncestor(pth,level=2))
            # BC_t
          else:
            refnode = parent
          (gridloc,rs)  = val_u.getGridLocation(refnode,pth=pth,log=log,rs=rs)
          if ( (gridloc==CGK.Vertex_s # Vertex data  
                or
                (gridloc.endswith(CGK.FaceCenter_s)
                 and ztype==CGK.Structured_s # Face data for structured grid
                 )
                  )
              and (len(self.context[CGK.VertexSize_s][zpth])==self.context[CGK.IndexDimension_s][zpth])
              ):
            dd=self.context[CGK.VertexSize_s][zpth]

            for d in range(self.context[CGK.IndexDimension_s][zpth]):
              if (   (pr[d][0]>dd[d])
                  or (pr[d][1]>dd[d])): # checking out of range (>VertexSize) 
                rs=log.push(pth,'S0206')
            if (    ztype==CGK.Structured_s
                and (    parent[3] in [CGK.GridConnectivity1to1_ts,CGK.BC_ts,CGK.BCDataSet_ts]
                     or (    parent[3]==CGK.GridConnectivity_ts 
                         and val_u.getGridConnectivityType(parent).startswith(CGK.Abutting_s))
                     )
                ): # checking pr defines a face (or less: edge or vertex)
              if (not val_u.isAFace(pr)): rs=log.push(pth,'S0219')
          elif (    (gridloc==CGK.CellCenter_s) # Cell center data 
                and (ztype==CGK.Structured_s) # for structured zone 
                and (len(self.context[CGK.CellSize_s][zpth])==self.context[CGK.IndexDimension_s][zpth])):
            dd=self.context[CGK.CellSize_s][zpth]
            for d in range(self.context[CGK.IndexDimension_s][zpth]):
              if (any(pr[d]>dd[d])):
                rs=log.push(pth,'S0211',CGK.CellCenter_s,d+1,self.context[CGK.IndexDimension_s][zpth],[1,dd[d]]) # checking out of range elements
          else : # Other GridLocation
            if (ztype==CGK.Unstructured_s): # Unstructured zone
              imin = 1
#              if (gridloc==CGK.FaceCenter_s): imin=self.context[CGK.CellSize_s][zpth][0]+1 # not in SIDS 
              imax=self.context[CGK.ElementsSize_s][zpth]
              if (any(pr.flatten()<imin) or any(pr.flatten()>imax)):
                rs=log.push(pth,'S0211',gridloc,1,self.context[CGK.IndexDimension_s][zpth],[imin,imax]) # checking out of range elements
              erl = val_u.getElementTypeRangeList(val_u.getLevelFromGridLoc(gridloc),
                                                  self.context[CGK.PhysicalDimension_s][zpth],
                                                  self.context[CGK.ElementRangeList_s][zpth])
              rs=val_u.allIndexInElementRangeList(pr[0],erl,pth,log,rs)               
        elif (node[0]==CGK.ElementRange_s): # ElementRange node
          if (ztype!=CGK.Unstructured_s or parent[3] != CGK.Elements_ts):
            rs=log.push(pth,'S0609') 
          else: 
            dd=self.context[CGK.ElementsSize_s][zpth]
            if (any(pr.flatten()>dd)):
               # checking out of range elements
              rs=log.push(pth,'S0217',1,
                          self.context[CGK.IndexDimension_s][zpth],[1,dd])
    return rs
  # --------------------------------------------------------------------
  def BC_t(self,pth,node,parent,tree,log):
    self.dbg("BC_t",pth)
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.BCType_l))):
      rs=log.push(pth,'S0108')
    if CGU.stringValueInList(node,[CGK.Null_s]):
      rs=log.push(pth,'S0221',CGK.Null_s)
    elif CGU.stringValueInList(node,[CGK.UserDefined_s]):
      rs=log.push(pth,'S0221',CGK.UserDefined_s)
    elif CGU.stringValueInList(node,[CGK.FamilySpecified_s]):
      if (not CGU.hasChildType(node,CGK.FamilyName_ts)):
        rs=log.push(pth,'S0304')
    if (not val_u.hasOneAndOnlyOneChildAmong(node,[CGK.PointList_s,CGK.PointRange_s])):
      rs=log.push(pth,'S0224',[CGK.PointList_s,CGK.PointRange_s]) # PointList or PointRange required
    physdim  = self.context[CGK.PhysicalDimension_s][pth]
    gridloc  = val_u.getGridLocation(node)
    agridloc = val_u.getAuthGridLocation(physdim,physdim-1)
    if (gridloc not in agridloc):# BC applies only on surface patch
      rs=log.push(pth,'S0218',gridloc,node[3],agridloc) 
    return rs
  # --------------------------------------------------------------------
  def ZoneBC_t(self,pth,node,parent,tree,log):
    self.dbg("ZoneBC_t",pth)
    rs=CGM.CHECK_OK
    if (not CGU.hasChildType(node,CGK.BC_ts)):
      rs=log.push(pth,'S0604')
    return rs
  # --------------------------------------------------------------------
  def FlowSolution_t(self,pth,node,parent,tree,log):
    self.dbg("FlowSolution_t",pth)
    rs=CGM.CHECK_OK
    (gridloc,rs)=val_u.getGridLocation(node,pth=pth,log=log,rs=rs)
    if (    val_u.getZoneType(tree,pth)==CGK.Unstructured_s # unstructured zone
        and not CGU.hasChildName(node,CGK.PointList_s)
        and not CGU.hasChildName(node,CGK.PointRange_s)
        and gridloc not in [CGK.Vertex_s,CGK.CellCenter_s]):
      rs=log.push(pth,'S0216',gridloc)
    datasize=val_u.dataSize(node,
                            self.context[CGK.IndexDimension_s][pth],
                            self.context[CGK.VertexSize_s][pth],
                            self.context[CGK.CellSize_s][pth])
    if (datasize):
      for n in CGU.hasChildType(node,CGK.DataArray_ts): 
        if (CGU.getShape(n)!=datasize): 
          rs=log.push(pth,'S0192',CGU.getShape(n),datasize,n[0])          
    return rs
  # --------------------------------------------------------------------
  def ZoneSubRegion_t(self,pth,node,parent,tree,log):
    self.dbg("ZoneSubRegion_t",pth)
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(1,)):  # check shape
      rs=log.push(pth,'S0192',CGU.getShape(node),(1,),node[0])
    celldim       = self.context[CGK.CellDimension_s][pth]
    regioncelldim = node[1][0]
    if (   regioncelldim > celldim
        or regioncelldim < 1): # check value
      rs=log.push(pth,'S0010')
    (gridloc,rs)  = val_u.getGridLocation(node,pth=pth,log=log,rs=rs)
    if (gridloc not in val_u.getAuthGridLocation(celldim,regioncelldim)): # check valid grid location
      rs=log.push(pth,'S0215',gridloc,celldim,regioncelldim) 
    children = [CGK.PointList_s,CGK.PointRange_s,CGK.BCRegionName_s,CGK.GridConnectivityRegionName_s]
    if (not val_u.hasOneAndOnlyOneChildAmong(node,children)): # check valid children  
      rs=log.push(pth,'S0224',children)
    listlength = -1
    if (CGU.hasChildName(node,CGK.PointList_s) or CGU.hasChildName(node,CGK.PointRange_s)):
      listlength = val_u.getListLength(node)
    if (CGU.hasChildName(node,CGK.BCRegionName_s)): # test if BC exists
      loc  = CGU.hasChildName(node,CGK.BCRegionName_s)
      test = False
      for path in CGU.getAllNodesByTypeSet(parent,CGK.BC_ts):
        n=CGU.getNodeByPath(parent,path)
        if (CGU.stringValueMatches(loc,n[0])): 
          test=True
          listlength = val_u.getListLength(n)
          break
      if (not test):
        rs=log.push(pth+'/'+CGK.BCRegionName_s,'S0010')
    if (CGU.hasChildName(node,CGK.GridConnectivityRegionName_s)): # test if GridConnectivity exists
      loc  = CGU.hasChildName(node,CGK.GridConnectivityRegionName_s)
      test = False
      for path in CGU.getAllNodesByTypeSet(parent,[CGK.GridConnectivity_ts,CGK.GridConnectivity1to1_ts]):
        n=CGU.getNodeByPath(parent,path)
        if (CGU.stringValueMatches(loc,n[0])): 
          test=True
          listlength = val_u.getListLength(n)          
          break
      if (not test):
        rs=log.push(pth+'/'+CGK.GridConnectivityRegionName_s,'S0010')
    if (listlength>0):
      for n in CGU.hasChildType(node,CGK.DataArray_ts): # check DataArray children shapes
        shp=(listlength,)
        if (CGU.getShape(n)!=shp): 
          rs=log.push(pth,'S0192',CGU.getShape(n),shp,n[0])
    return rs
  # --------------------------------------------------------------------
  def Rind_t(self,pth,node,parent,tree,log):
    self.dbg("Rind_t",pth)
    rs=CGM.CHECK_OK
    shp=(2*self.context[CGK.IndexDimension_s][pth],)
    if (CGU.getShape(node)!=shp):  # check shape
      rs=log.push(pth,'S0192',CGU.getShape(node),shp,node[0])
    ntype = CGU.getValueDataType(node)
    if ((CGU.getAuthDataTypes(node) is not None) 
        and (ntype not in CGU.getAuthDataTypes(node))): # checking date type 
      rs=log.push(pth,'S0199',ntype,CGU.getAuthDataTypes(node))
    return rs
  # --------------------------------------------------------------------
  def Descriptor_t(self,pth,node,parent,tree,log):
    self.dbg("Descriptor_t",pth)
    rs=CGM.CHECK_OK
    if (node[1] is not None):
      v=set(node[1])
      p=set(string.printable)
      if (not v.issubset(p)):
        rs=log.push(pth,'S0270')
    else:
        rs=log.push(pth,'S0271')      
    return rs
  # --------------------------------------------------------------------
  def FamilyBCDataSet_t(self,pth,node,parent,tree,log):
    self.dbg("FamilyBCDataSet_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def FamilyBC_t(self,pth,node,parent,tree,log):
    self.dbg("FamilyBC_t",pth)
    rs=CGM.CHECK_OK
    if (node[0]!=CGK.FamilyBC_s):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.C1])
    if (CGU.getValueDataType(node) not in [CGK.C1]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.C1])
    elif (not CGU.stringValueInList(node,CGK.BCType_l)):
      rs=log.push(pth,'S0108')
    elif CGU.stringValueInList(node,[CGK.Null_s]):
      rs=log.push(pth,'S0221',CGK.Null_s)
    elif CGU.stringValueInList(node,[CGK.UserDefined_s]):
      rs=log.push(pth,'S0221',CGK.UserDefined_s)
    return rs
  # --------------------------------------------------------------------
  def FamilyName_t(self,pth,node,parent,tree,log):
    self.dbg("FamilyName_t",pth)
    rs=CGM.CHECK_OK
    famdefinition=CGU.getNodeByPath(tree,pth)
    if (famdefinition[1] is None):
      rs=log.push(pth,'S0300')
    else:
      famtarget=famdefinition[1].tostring().rstrip()
      basename=val_u.getBase(pth,node,parent,tree,log)
      famtargetpath="/%s/%s"%(basename,famtarget)
      if (famtargetpath not in self.context):
        famtargetnode=CGU.getNodeByPath(tree,famtargetpath)
        if (famtargetnode is None):
          rs=log.push(pth,'S0301',famtarget)
        else:
          self.context[famtargetpath][pth]=True
    return rs
  # --------------------------------------------------------------------
  def ConvergenceHistory_t(self,pth,node,parent,tree,log):
    self.dbg("ConvergenceHistory_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def ZoneGridConnectivity_t(self,pth,node,parent,tree,log):
    self.dbg("ZoneGridConnectivity_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def GridConnectivity1to1_t(self,pth,node,parent,tree,log):
    self.dbg("GridConnectivity1to1_t",pth)
    rs=CGM.CHECK_OK
    if (val_u.getZoneType(tree,pth)!=CGK.Structured_s):
      rs=log.push(pth,'S0620')
    er =CGU.hasChildName(node,CGK.PointRange_s)
    erd=CGU.hasChildName(node,CGK.PointRangeDonor_s)
    if (not er):
      rs=log.push(pth,'S0194',CGK.PointRange_s)   
    if (not erd):
      rs=log.push(pth,'S0194',CGK.PointRangeDonor_s)      
    ntype = CGU.getValueDataType(node)
    if ((CGU.getAuthDataTypes(node) is not None) 
      and (ntype not in CGU.getAuthDataTypes(node))): # checking date type 
      rs=log.push(pth,'S0199',ntype,CGU.getAuthDataTypes(node))
    else: # checking data value, i.e valid ZoneDonorName
      zdonorname=node[1].tostring().rstrip()
      basename=val_u.getBase(pth,node,parent,tree,log)
      targetpath='/%s/%s'%(basename,zdonorname)
      zdonor=CGU.getNodeByPath(tree,targetpath)
      if (zdonor is None): rs=log.push(pth,'S0401',targetpath)
      elif  (val_u.getZoneType(tree,targetpath)!=CGK.Structured_s):  # checking donor zone is structured
        rs=log.push(pth,'S0620')
      else: # look for same node on donor zone side
        found=False
        for path in CGU.getAllNodesByTypeSet(zdonor,[CGK.GridConnectivity1to1_ts]):
          n=CGU.getNodeByPath(zdonor,path)
          if (CGU.stringValueMatches(n,val_u.getZoneName(tree,pth))):
            der =CGU.hasChildName(n,CGK.PointRange_s)
            derd=CGU.hasChildName(n,CGK.PointRangeDonor_s)
            if (    der and derd and er and erd
                and CGU.checkSameValue(der,erd) 
                and CGU.checkSameValue(derd,er)): # Note: compare PointRange and PointRangeDonor, should compare Transpose instead
              found=True
              break
        if (not found):
          rs=log.push(pth,'S0621',zdonorname)
    (gridloc,rs) = val_u.getGridLocation(node,pth=pth,log=log,rs=rs)
    if (gridloc != CGK.Vertex_s): # checking grid location is vertex
      rs=log.push(pth,'S0218',gridloc,node[3],[CGK.Vertex_s])
    return rs
  # --------------------------------------------------------------------
  def GridConnectivity_t(self,pth,node,parent,tree,log):
    self.dbg("GridConnectivity_t",pth)
    rs=CGM.CHECK_OK
    ntype = CGU.getValueDataType(node)
    if ((CGU.getAuthDataTypes(node) is not None) 
      and (ntype not in CGU.getAuthDataTypes(node))): # checking date type 
      rs=log.push(pth,'S0199',ntype,CGU.getAuthDataTypes(node))
    else: # checking data value, i.e valid ZoneDonorName
      zdonorname=node[1].tostring().rstrip()
      basename=val_u.getBase(pth,node,parent,tree,log)
      targetpath='/%s/%s'%(basename,zdonorname)
      zdonor=CGU.getNodeByPath(tree,targetpath)
      if (zdonor is None): rs=log.push(pth,'S0401',targetpath)
    # checking has one and only one of those children: PointList, PointRange
    if (not val_u.hasOneAndOnlyOneChildAmong(node,[CGK.PointList_s,CGK.PointRange_s])):
      rs=log.push(pth,'S0224',[CGK.PointList_s,CGK.PointRange_s])  
    # checking coherent GridLocation and GridConnectivityType      
    (gridloc,rs) = val_u.getGridLocation(node,pth=pth,log=log,rs=rs)
    (gridct,rs)  = val_u.getGridConnectivityType(node,pth=pth,log=log,rs=rs)
    if (    gridct in [CGK.Abutting_s,CGK.Abutting1to1_s]
        and not (   gridloc == CGK.Vertex_s
                 or gridloc.endswith(CGK.FaceCenter_s))):
      rs=log.push(pth,'S0218',gridloc,gridct,[CGK.Vertex_s,CGK.FaceCenter_s,
                                             CGK.IFaceCenter_s,CGK.JFaceCenter_s,CGK.KFaceCenter_s])
    elif (    gridct      in [CGK.Overset_s]
          and gridloc not in [CGK.Vertex_s,CGK.CellCenter_s]):
      rs=log.push(pth,'S0218',gridloc,gridct,[CGK.Vertex_s,CGK.CellCenter_s])
    return rs
  # --------------------------------------------------------------------
  def GridCoordinates_t(self,pth,node,parent,tree,log):
    self.dbg("GridCoordinates_t",pth)
    rs=CGM.CHECK_OK
    cl=CGU.hasChildType(node,CGK.DataArray_ts)
    if (not cl):
      rs=log.push(pth,'S0260')
      return rs
    if (len(cl)!=self.context[CGK.PhysicalDimension_s][pth]):
      rs=log.push(pth,'S0261')
    cn=set()
    datasize=val_u.dataSize(node,
                            self.context[CGK.IndexDimension_s][pth],
                            self.context[CGK.VertexSize_s][pth],
                            self.context[CGK.CellSize_s][pth])
    for c in cl:
      if (c[1] is not None):
        if (CGU.getShape(c)!=datasize):
          rs=log.push(pth,'S0262')
      cn.add(c[0])
    if   (self.context[CGK.PhysicalDimension_s][pth]==1):
      if (not cn.issubset(set((CGK.CoordinateX_s,
                               CGK.CoordinateR_s,
                               CGK.CoordinateXi_s)))):
        rs=log.push(pth,'S0264')
    elif (self.context[CGK.PhysicalDimension_s][pth]==2):
      if (not cn.issubset(set((CGK.CoordinateX_s,
                               CGK.CoordinateR_s,
                               CGK.CoordinateXi_s,
                               CGK.CoordinateY_s,
                               CGK.CoordinateTheta_s,
                               CGK.CoordinateEta_s)))):
        rs=log.push(pth,'S0265')
      if ((CGK.CoordinateX_s in cn) and (CGK.CoordinateY_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateR_s in cn) and (CGK.CoordinateTheta_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateXi_s in cn) and (CGK.CoordinateEta_s not in cn)):
        rs=log.push(pth,'S0263')
    else:
      if (not cn.issubset(set((CGK.CoordinateX_s,
                               CGK.CoordinateR_s,
                               CGK.CoordinateXi_s,
                               CGK.CoordinateY_s,
                               CGK.CoordinateTheta_s,
                               CGK.CoordinateEta_s,
                               CGK.CoordinateZ_s,
                               CGK.CoordinatePhi_s,
                               CGK.CoordinateZeta_s)))):
        rs=log.push(pth,'S0265')
      if ((CGK.CoordinateX_s in cn) and (CGK.CoordinateY_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateX_s in cn) and (CGK.CoordinateZ_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateR_s in cn) and (CGK.CoordinateTheta_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateR_s in cn) and (CGK.CoordinatePhi_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateXi_s in cn) and (CGK.CoordinateEta_s not in cn)):
        rs=log.push(pth,'S0263')
      elif ((CGK.CoordinateXi_s in cn) and (CGK.CoordinateZeta_s not in cn)):
        rs=log.push(pth,'S0263')
    return rs
  # --------------------------------------------------------------------
  def UserDefinedData_t(self,pth,node,parent,tree,log):
    self.dbg("UserDefinedData_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def Axisymmetry_t(self,pth,node,parent,tree,log):
    self.dbg("Axisymmetry_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def FlowEquationSet_t(self,pth,node,parent,tree,log):
    self.dbg("FlowEquationSet_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def Gravity_t(self,pth,node,parent,tree,log):
    self.dbg("Gravity_t",pth)
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def RotatingCoordinates_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def ChemicalKineticsModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.ChemicalKineticsModelType_l))):
      rs=log.push(pth,'S0183')
    return rs
  # --------------------------------------------------------------------
  def EMConductivityModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.EMConductivityModelType_l))):
      rs=log.push(pth,'S0187')
    return rs
  # --------------------------------------------------------------------
  def EMElectricFieldModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.EMElectricFieldModelType_l))):
      rs=log.push(pth,'S0185')
    return rs
  # --------------------------------------------------------------------
  def ViscosityModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.ViscosityModelType_l))):
      rs=log.push(pth,'S0178')
    return rs
  # --------------------------------------------------------------------
  def TurbulenceClosure_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.TurbulenceClosureType_l))):
      rs=log.push(pth,'S0174')
    return rs
  # --------------------------------------------------------------------
  def ArbitraryGridMotion_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.ArbitraryGridMotionType_l))):
      rs=log.push(pth,'S0118')
    return rs
  # --------------------------------------------------------------------
  def Periodic_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def BCData_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (node[0] not in CGK.BCDataType_l):
      rs=log.push(pth,'S0106')
    return rs
  # --------------------------------------------------------------------
  def GridConnectivityProperty_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def BCDataSet_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (    CGU.hasChildName(node,CGK.PointList_s) 
        and CGU.hasChildName(node,CGK.PointRange_s)):
      rs=log.push(pth,'S0222')
    physdim  = self.context[CGK.PhysicalDimension_s][pth]
    defaultv = val_u.getGridLocation(parent) # if gridloc absent, value taken at upper level (BC_t)
    (gridloc,rs)  = val_u.getChildNodeValueWithDefault(node,CGK.GridLocation_ts,defaultv,pth=pth,log=log,rs=rs)    
    agridloc = val_u.getAuthGridLocation(physdim,physdim-1)
    if (gridloc not in agridloc):# BC applies only on surface patch
      rs=log.push(pth,'S0218',gridloc,CGK.BCDataSet_ts,agridloc)      
    return rs
  # --------------------------------------------------------------------
  def TurbulenceModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.TurbulenceModelType_l))):
      rs=log.push(pth,'S0177')
    return rs
  # --------------------------------------------------------------------
  def OversetHoles_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def ThermalRelaxationModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.ThermalRelaxationModelType_l))):
      rs=log.push(pth,'S0182')
    return rs
  # --------------------------------------------------------------------
  def ThermalConductivityModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None) or
        (not CGU.stringValueInList(node,CGK.ThermalConductivityModelType_l))):
      rs=log.push(pth,'S0173')
    return rs
  # --------------------------------------------------------------------
  def GoverningEquations_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.GoverningEquationsType_l))):
      rs=log.push(pth,'S0172')
    return rs
  # --------------------------------------------------------------------
  def GasModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.GasModelType_l))):
      rs=log.push(pth,'S0180')
    return rs
  # --------------------------------------------------------------------
  def DiffusionModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (node[1] is None):
      rs=log.push(pth,'S0280')
    elif (CGU.getShape(node)[0] not in (1,3,6)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.I4]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.I4])
    else:
      cd=self.context[CGK.CellDimension_s][pth]
      if (    ((cd==1) and (CGU.getShape(node)[0]!=1))
           or ((cd==2) and (CGU.getShape(node)[0]!=3))
           or ((cd==3) and (CGU.getShape(node)[0]!=6))):
        rs=log.push(pth,'S0281',cd)
      else:
        v=set(node[1].flat)
        if (not v.issubset(set((0,1)))):
              rs=log.push(pth,'S0282')
    return rs
  # --------------------------------------------------------------------
  def EMMagneticFieldModel_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.EMMagneticFieldModelType_l))):
      rs=log.push(pth,'S0186')
    return rs
  # --------------------------------------------------------------------
  def IntegralData_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def Ordinal_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    return rs
  # --------------------------------------------------------------------
  def DimensionalUnits_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(32,5)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.C1]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.I4])
    else:
      d=node[1].T
      tname=d[0].tostring().rstrip()
      if (tname not in CGK.MassUnits_l):        rs=log.push(pth,'S0110')
      tname=d[1].tostring().rstrip()
      if (tname not in CGK.LengthUnits_l):      rs=log.push(pth,'S0111')
      tname=d[2].tostring().rstrip()
      if (tname not in CGK.TimeUnits_l):        rs=log.push(pth,'S0112')
      tname=d[3].tostring().rstrip()
      if (tname not in CGK.TemperatureUnits_l): rs=log.push(pth,'S0113')
      tname=d[4].tostring().rstrip()
      if (tname not in CGK.AngleUnits_l):       rs=log.push(pth,'S0114')
    dc=CGU.hasChildName(parent,CGK.DataClass_s)
    if (dc is None): rs=log.push(pth,'S0164')
    return rs
  # --------------------------------------------------------------------
  def AdditionalUnits_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(32,3)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.C1]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.I4])
    else:
      d=node[1].T
      tname=d[0].tostring().rstrip()
      if (tname not in CGK.ElectricCurrentUnits_l):   rs=log.push(pth,'S0115')
      tname=d[1].tostring().rstrip()
      if (tname not in CGK.SubstanceAmountUnits_l):   rs=log.push(pth,'S0116')
      tname=d[2].tostring().rstrip()
      if (tname not in CGK.LuminousIntensityUnits_l): rs=log.push(pth,'S0117')
    return rs
  # --------------------------------------------------------------------
  def DimensionalExponents_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(5,)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.R4, CGK.R8]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.R4, CGK.R8])
    return rs
  # --------------------------------------------------------------------
  def AdditionalExponents_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(3,)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.R4, CGK.R8]):
      rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.R4, CGK.R8])
    return rs
  # --------------------------------------------------------------------
  def DataConversion_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,'S0191')
    return rs
  # --------------------------------------------------------------------
  def DataClass_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    du=CGU.hasChildName(parent,CGK.DimensionalUnits_s)
    if (du is None):
      if (self.context[CGK.DimensionalUnits_s][pth] is not None):
        rs=log.push(pth,'S0154')
      else:
        rs=log.push(pth,'S0153')
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.DataClass_l))):
      rs=log.push(pth,'S0105')
    elif ((du is None)
           and CGU.stringValueInList(node,[CGK.Dimensional_s,
                                           CGK.NormalizedByDimensional_s])):
        rs=log.push(pth,'S0161')
    elif ((du is not None)
          and CGU.stringValueInList(node,[CGK.NormalizedByUnknownDimensional_s,
                                          CGK.DimensionlessConstant_s,
                                          CGK.NondimensionalParameter_s])):
        rs=log.push(pth,'S0162')
    return rs
  # --------------------------------------------------------------------
  def DataArray_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    dtc=CGU.hasChildName(node,CGK.DataClass_s)
    if (dtc is None):
      dtc=self.context[CGK.DataClass_s][pth]
      if (dtc is None):
        rs=log.push(pth,'S0150')
      else:
        rs=log.push(pth,'S0155')        
    name=node[0]
    if (name not in CGK.cgnsnames):
      rs=log.push(pth,'S0121',name)
    du=CGU.hasChildName(node,CGK.DimensionalUnits_s)      
    de=CGU.hasChildName(node,CGK.DimensionalExponents_s)
    dc=CGU.hasChildName(node,CGK.DataConversion_s)
    if ((dtc is not None) and (du is None)):
      if (self.context[CGK.DimensionalUnits_s][pth] is not None):
        rs=log.push(pth,'S0154')
      else:
        rs=log.push(pth,'S0153')
    if ((de is not None) and (dtc is None)): rs=log.push(pth,'S0156')
    if ((de is not None) and (du is None)):  rs=log.push(pth,'S0157')
    if ((dc is not None) and (dtc is None)): rs=log.push(pth,'S0158')
    if ((dc is not None) and (du is None)):  rs=log.push(pth,'S0159')
    if ((dc is not None) and (de is None)):  rs=log.push(pth,'S0160')
    if (dtc is not None):
      if ((de is not None)
           and CGU.stringValueInList(dtc,[CGK.NormalizedByUnknownDimensional_s,
                                          CGK.DimensionlessConstant_s,
                                          CGK.NondimensionalParameter_s])):
        rs=log.push(pth,'S0163')
    return rs
  # --------------------------------------------------------------------
  def GridConnectivityType_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridConnectivityType_l)):
      rs=log.push(pth,'S0104')
    return rs
  # --------------------------------------------------------------------
  def GridLocation_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (not CGU.stringValueInList(node,CGK.GridLocation_l)):
      rs=log.push(pth,'S0103')
    return rs
  # --------------------------------------------------------------------
  def IntIndexDimension_ts(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    idim = self.context[CGK.IndexDimension_s][pth]
    if (node[0]==CGK.Transform_s):
      if (CGU.getShape(node)!=(idim,)):
        rs=log.push(pth,'S0192',CGU.getShape(node),(idim,),node[0])
      elif (CGU.getValueDataType(node) not in [CGK.I4,CGK.I8]):
        rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.I4,CGK.I8])
      else:
        tr=list(node[1].flat)
        if (not CGS.transformCheckValues(tr,self.context[CGK.CellDimension_s][pth])):
          rs=log.push(pth,'S0204')
    elif (node[0]==CGK.InwardNormalIndex_s):
      if (CGU.getShape(node)!=(idim,)): # checking shape
        rs=log.push(pth,'S0192',CGU.getShape(node),(idim,),node[0])
      elif (CGU.getValueDataType(node) not in [CGK.I4,CGK.I8]): # checking data type
        rs=log.push(pth,'S0199',CGU.getValueDataType(node),[CGK.I4,CGK.I8])
      else: 
        if (   (any(abs(node[1]) > 1))
            or (NPY.where(node[1] != 0)[0].size != 1)):  # checking data values
          rs=log.push(pth,'S0230',node[0],'among [-1,0,+1], only one nonzero element')
      zname=val_u.getZoneName(tree,pth)
      basename=val_u.getBase(pth,node,parent,tree,log)
      zpath='/%s/%s'%(basename,zname)
      zdonor=CGU.getNodeByPath(tree,zpath)
      if (zdonor is not None): 
        ztype = val_u.getZoneType(tree,zpath)
        if (ztype != CGK.Structured_s):  # no sense unless zone is structured
          rs=log.push(pth,'S0710',ztype)          
    return rs
  # --------------------------------------------------------------------
  def Elements_t(self,pth,node,parent,tree,log):
    
    # --------------------------------------------------------------------
    def getElementDataSize(etype,erange,econnect,estartoffset,pth,log,rs):
      # calculate number of data in econnectivity, function of element type and range; return
      # list of element1,element2, depending on type of Elements
      cdef int i,er_start,er_end,ndata,elementdatasize,elementsize,index,ecsize,eosize,j,l
      cdef NCY.ndarray econnectarray,eoffsetarray,etypearray,element1,element2
      cdef int *ecdata
      cdef int *eoffsetdata
      cdef int *etdata
      cdef int *el1data
      cdef int *el2data
      
      etypearray    = NPY.array(CGK.ElementTypeNPE_l,dtype='int32')
      econnectarray = econnect[1]
      ecdata        = <int*>econnectarray.data
      etdata        = <int*>etypearray.data

      er_start    = min(erange[1][0]) # ElementRange start and end
      er_end      = max(erange[1][0])
      elementsize = er_end-er_start+1 # number of elements
      ecsize      = econnect[1].size
      restofsize  = ecsize-elementsize
      
      elementdatasize = 0 # initialize
      element1        = NPY.empty(elementsize,'i',order='F') # initialize
      element2        = NPY.empty(restofsize,'i',order='F') # initialize
      el1data         = <int*>element1.data
      el2data         = <int*>element2.data
      
      if etype in [CGK.NGON_n,CGK.NFACE_n]:
         eoffsetarray = estartoffset[1]
         eoffsetdata  = <int*>eoffsetarray.data
         eosize      = estartoffset[1].size

      if (not CGU.checkArrayInteger(econnectarray)): # check all data in ElementConnectivity are integers (note: even for MIXED, data are integers since ElementType are described through integers)
        rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0004')
        return rs,-1,[],[]

      if (etype == CGK.MIXED): # In case of MIXED elements
        index = 0
        for i in range(er_start,er_end+1):
          if (index<=ecsize-1): # check ElementConnectivity data long enough to reach index
            l = len(CGK.ElementType)
            if (ecdata[index]<0 or ecdata[index]>l):
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0109')
              return rs,-1,[],[]
            else:
              ndata = etdata[ecdata[index]] # number of data for the current element
          else:
            rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0191')
            return rs,-1,[],[]       
          elementdatasize += 1+ndata
          el1data[i-er_start]=ecdata[index] # List of ElementType
          if (index-(i-er_start)+ndata>restofsize):
            rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0191')
            return rs,-1,[],[]    
          else:
            for j in range(index+1,index+ndata+1):
              el2data[j-(i-er_start)-1]=ecdata[j] # List of node index              
#          element2=NPY.concatenate((element2,econnect[1][index+1:index+ndata+1])) # List of node index
          index           += 1+ndata           
      elif (etype in [CGK.NGON_n,CGK.NFACE_n]): # in case of NGON or NFACE elements
        index = 0
        if (elementsize+1>eosize):
            rs=log.push(pth+'/'+CGK.ElementStartOffset_s,'S0191')
            return rs,-1,[],[]
        for i in range(er_start,er_end+1):
          ndata = eoffsetdata[i+1-er_start]-eoffsetdata[i-er_start]
          if not (index<=ecsize-1): # check ElementConnectivity data not long enough
            rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0191')
            return rs,-1,[],[]
          elementdatasize += ndata
          el1data[i-er_start]= ndata # List of ElementType
          if (index+ndata>ecsize):
            rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0191')
            return rs,-1,[],[]
          else:
            for j in range(index,index+ndata):
              el2data[j-(i-er_start)]=ecdata[j] # List of node (NGON) or face (NFACE) indexes
#          element2=NPY.concatenate((element2,econnect[1][index+1:index+ndata+1])) # List of node (NGON) or face (NFACE) indexes
          index           += ndata
      else :
        elementdatasize = elementsize*CGK.ElementTypeNPE_l[etype]
        element1        = econnect[1]
      return rs,elementdatasize,element1,element2
    
    # --------------------------------------------------------------------
    def checkParentElements(penode,int et,pth,log,rs,ppnode=None):
      """
        - Args:
         * `penode`: ParentElements node
         * `ppnode`: ParentElementsPosition node
         * `ppnode`: if given, ppnode is treated and not penode, else, penode is treated
         * `et`: ElementType of face Elements, parent node of pnode and penode
         * `pth`: face Elements node path, parent node of pnode and penode         
      """

      cdef int i,j,erange,jrange,ejrange,laetr,laet,var,etcell,etp
      cdef NCY.ndarray pearray,pparray,aetarray

      (erange,jrange) = CGU.getShape(penode) # (ElementRange dimension, 2)
      if (jrange != 2): return rs
      ejrange = erange*jrange 
      aet = val_u.getAuthElementTypes(et) # authorized element types for the face   
      if (not aet):     return rs
      aetarray = NPY.array(aet,dtype='int32')
      laet     = len(aetarray)        
      pearray = penode[1].flatten()
      if (ppnode is not None):
        pparray = ppnode[1].flatten()
      erl = self.context[CGK.ElementRangeList_s][pth]
      for i in range(ejrange):
        etp  = -1 # initialize parent element type at incoherent value
        var  = pearray[i]
        if (var==0): continue  # 0 authorized for boundary faces, nothing to check
        for j in range(laet):
          etcell = aetarray[j] # authorized cell type for the face
          if (val_u.indexInRangeList(var,erl[etcell])): # the cell attached to the face has been found in ElementRangeList of authorized cells
            etp=etcell # save the cell type
            break
        if (etp == -1): # the cell attached to the face has not been found in any ElementRange of authorized cells
          rs=log.push(pth+'/'+CGK.ParentElements_s,'S0212',[int(i/2),i%2],CGK.ElementType_[et]) # checking not authorized cell type for the current face type
          return rs
        elif (etp != -1 and ppnode is not None):
          var = pparray[i]
          if (    (etp in CGK.ElementType_tetra and var not in range(1,4+1))
               or (etp in CGK.ElementType_pyra  and var not in range(1,5+1))
               or (etp in CGK.ElementType_penta and var not in range(1,5+1))
               or (etp in CGK.ElementType_hexa  and var not in range(1,6+1))):
            rs=log.push(pth+'/'+CGK.ParentElementsPosition_s,'S0213',CGK.ElementType_[etp],var,[int(i/2),i%2]) # bad face position for this element type
            return rs
      return rs

    # --------------------------------------------------------------------
    def getBndFacesFromParentElements(penode):
      """
        - Args:
         * `penode`: ParentElements node
      """
      cdef int i,erange,jrange,ejrange
      cdef NCY.ndarray pearray

      (erange,jrange) = CGU.getShape(penode) # (ElementRange dimension, 2)
      if (jrange != 2): return None
      ejrange = erange*jrange 
      pearray = penode[1].flatten()
      bndfaces = []
      for i in range(ejrange):
        if (pearray[i]==0): bndfaces.append(int(i/2))
      return list(set(bndfaces))

    # --------------------------------------------------------------------
    def checkAllBndFacesOnBCOrGridConnect(bndfaces,method,tree,pth,log,rs):
      """
        - Args:
         * `bndfaces`: list of boundary faces numbers, local indices for current Elements_t, not global for current zone
         * `method`: node name or parameter used to determine bndfaces 
         * `tree`: complete CGNS tree (at least, zone)
         * `pth`: path to face Elements_t node
      """
      node = CGU.getNodeByPath(tree,pth) # Elements_t node
      er     = CGU.hasChildName(node,CGK.ElementRange_s) # Element range
      if (CGU.getShape(er)==(self.context[CGK.IndexDimension_s][pth],2)):
        # zone path
        zonepath = CGU.getPathAncestor(pth)
        # get IndexRange and IndexArray on all BC and GridConnectivity in zone
        (rl,al) = val_u.getIndicesOnBCandGC(tree,zonepath,self.context[CGK.IndexDimension_s][zonepath])
        absent    = [] # initialise tables of faces not on BC or GC
        duplicate = [] # initialise tables of faces on several BC and/or GC
        # check on boundary faces
        idx_min=min(er[1][0])
        for i in bndfaces:
          idx=idx_min+i
          n = val_u.countIdxInRangeList(idx,rl)
          n += NPY.where(al.flatten() == idx)[0].size
          if  (n == 0): absent.append(idx)
          elif (n > 1): duplicate.append(idx)
        if (absent):    rs=log.push(pth,'S0700',absent,method)
        if (duplicate): rs=log.push(pth,'S0701',duplicate,method)
      return rs
        
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(2,)):
      rs=log.push(pth,'S0191')
    else:
      et=node[1][0]
      eb=node[1][1]
      self.context[CGK.ElementType_s][pth]=et
      self.context[CGK.ElementSizeBoundary_s][pth]=eb
      if (et not in range(0,len(CGK.ElementType)+1)):
        rs=log.push(pth,'S0109')
        bad_et = True
      else: bad_et = False
      if (eb==0): bad_eb=False
      elif (eb<0): bad_eb=True
      er = CGU.hasChildName(node,CGK.ElementRange_s)    
      ec = CGU.hasChildName(node,CGK.ElementConnectivity_s)
      eo = CGU.hasChildName(node,CGK.ElementStartOffset_s)
      if (er is None) : 
        rs=log.push(pth,'S0607') # checking if ElementRange node is present;
        bad_er = True
      else:
        if (CGU.getShape(er)==(self.context[CGK.IndexDimension_s][pth],2)):
          bad_er = False
          elementsize=max(er[1][0])-min(er[1][0])+1
          if (eb>elementsize): bad_eb=True
          else: bad_eb=False
          shp_p=(elementsize,2) # shape for ParentElements and ParentElementsPosition nodes
        else: #Error for bad ElementRange node shape already raised in IndexRange_t
          bad_er = True
      if (bad_eb): rs=log.push(pth,'S0205') # bad ElementSizeBoundary
      elif ((not bad_eb) and (not bad_er) and (eb != 0)):
        rs=checkAllBndFacesOnBCOrGridConnect(range(eb),CGK.ElementSizeBoundary_s,tree,pth,log,rs)
      if (ec is None) : rs=log.push(pth,'S0606') # checking if ElementConnectivity node is present
      if ((eo is None) and (et in [CGK.NGON_n,CGK.NFACE_n])):
         rs=log.push(pth, 'S0610')# checking if ElementStartOffset node is present
         ec=None # Force ec to None since connectivity can not be valid without offset
      if ((ec is not None) and (not bad_et) and (not bad_er)):
        (rs,elementdatasize,element1,element2)=getElementDataSize(et,er,ec,pth,log,rs) # calculating ElementConnectivity data length
        rs=val_u.checkChildValue(ec,(elementdatasize,),[CGK.I4,CGK.I8],pth,log,rs) # checking ElementConnectivity node shape and type (note: even for MIXED, data are integers since ElementType are described through integers)
        if ((CGU.getValueDataType(ec) in [CGK.I4,CGK.I8]) 
            and (len(self.context[CGK.VertexSize_s][pth])==1)
            and elementdatasize!=-1):
          if (et == CGK.MIXED):
            # data of ElementConnectivity are Node or Etype indexes
            if (any(element1 > len(CGK.ElementType))): # checking out of range elements
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') 
            if (any(element2 > self.context[CGK.VertexSize_s][pth][0])): # checking out of range elements
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') 
          elif (et == CGK.NGON_n):
            # data of ElementConnectivity are Node index or Nnode by face
            if (any(element1 < 1)): # checking out of range elements
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') 
            if (any(element2 > self.context[CGK.VertexSize_s][pth][0])): # checking out of range elements
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') 
          elif (et == CGK.NFACE_n):
            # data of ElementConnectivity are Face index or Nface
            # catch range indices for faces, creates face range list (frl)
            frl = val_u.getElementTypeRangeList(CGK.Face_s,
                                                self.context[CGK.PhysicalDimension_s][pth],
                                                self.context[CGK.ElementRangeList_s][pth])
            if (any(element1 < 1)): # checking out of range elements
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') 
            if (any(abs(element2) <  1)): # checking out of range elements (Note: faces are oriented with sign - or + -> abs() needed)
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206')
            rs=val_u.allIndexInElementRangeList(abs(element2),frl,pth+'/'+CGK.ElementConnectivity_s,log,rs)
          else:
            # data of ElementConnectivity are Node index
            if (any(ec[1] > self.context[CGK.VertexSize_s][pth][0])): 
              rs=log.push(pth+'/'+CGK.ElementConnectivity_s,'S0206') # checking out of range elements
      pe = CGU.hasChildName(node,CGK.ParentElements_s)
      pp = CGU.hasChildName(node,CGK.ParentElementsPosition_s)
      if ((pe is not None) and (not bad_et)): # ParentElements are provided
        if (et not in val_u.getElementTypes(CGK.Face_s,self.context[CGK.PhysicalDimension_s][pth])):
          rs=log.push(pth+'/'+CGK.ParentElements_s,'S0210',CGK.ParentElements_s) # child authorized on face Elements only
        if (not bad_er):
          rs=val_u.checkChildValue(pe,shp_p,[CGK.I4,CGK.I8],pth,log,rs) # checking ParentElements node shape and type
          if (len(self.context[CGK.CellSize_s][pth])==self.context[CGK.IndexDimension_s][pth]): # if CellSize is defined
            cellsize = self.context[CGK.CellSize_s][pth][0]
            if (   any(pe[1].flatten()>cellsize) 
                or any(pe[1].flatten()<0       )): # 0 used for boundary faces
              rs=log.push(pth+'/'+CGK.ParentElements_s,'S0206') # checking out of range elements
            else:
              maxpe = NPY.array([max(pei) for pei in pe[1]])
              if (maxpe.min() == 0): # has tracked prohibited (0,0) adjacent cell couples
                rs=log.push(pth+'/'+CGK.ParentElements_s,'S0220',NPY.where(maxpe == 0)[0]+min(er[1][0]))
              if (not bad_eb and (eb != 0)):
                minpe = NPY.array([min(pei) for pei in pe[1][:eb]])
                if (any(minpe != 0)): # has tracked elements declared as boundary faces regarding ElementSizeBoundary that have no adjacent cell declared 0 in ParentElements
                  rs=log.push(pth+'/'+CGK.ParentElements_s,'S0704',NPY.where(minpe != 0)[0]+min(er[1][0]))
              bndfaces = getBndFacesFromParentElements(pe)
              rs=checkAllBndFacesOnBCOrGridConnect(bndfaces,CGK.ParentElements_s,tree,pth,log,rs)
          if (pp is None): rs=checkParentElements(pe,et,pth,log,rs)  # checking only ParentElements
      if ((pp is not None) and (not bad_et)): # ParentElementsPosition are provided
        if (et not in val_u.getElementTypes(CGK.Face_s,self.context[CGK.PhysicalDimension_s][pth])):
          rs=log.push(pth+'/'+CGK.ParentElementsPosition_s,'S0210',CGK.ParentElementsPosition_s) # child authorized on face Elements only
        if (not bad_er):
          rs=val_u.checkChildValue(pp,shp_p,[CGK.I4,CGK.I8],pth,log,rs) # checking ParentElementsPosition node shape and type
          if (pe is not None):       
            if (    (CGU.getShape(pp)==CGU.getShape(pe))
                and len(CGU.getShape(pp))==2):
              rs=checkParentElements(pe,et,pth,log,rs,ppnode=pp) # checking both ParentElements and ParentElementsPosition
          else:
            rs=log.push(pth,'S0214')
    return rs
  # --------------------------------------------------------------------
  def BaseIterativeData_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    if (CGU.getShape(node)!=(1,)):
      rs=log.push(pth,'S0191')
    elif (CGU.getValueDataType(node) not in [CGK.I4, CGK.I8]):
      rs=log.push(pth,'S0004')
    elif (node[1][0] < 0):
      rs=log.push(pth,'S0502',node[1][0])
    else:
      self.context[CGK.NumberOfSteps_s][pth]=node[1][0]
    itv=CGU.hasChildName(node,CGK.IterationValues_s)
    tmv=CGU.hasChildName(node,CGK.TimeValues_s)
    fnb=CGU.hasChildName(node,CGK.NumberOfFamilies_s)
    znb=CGU.hasChildName(node,CGK.NumberOfZones_s)
    fpt=CGU.hasChildName(node,CGK.FamilyPointers_s)
    zpt=CGU.hasChildName(node,CGK.ZonePointers_s)
    if (fnb is None and fpt is not None):
      rs=log.push(pth,'S0194',CGK.NumberOfFamilies_s)
    if (fnb is not None and fpt is None):
      rs=log.push(pth,'S0194',CGK.FamilyPointers_s)
    if (znb is None and zpt is not None):
      rs=log.push(pth,'S0194',CGK.NumberOfZones_s)
    if (znb is not None and zpt is None):
      rs=log.push(pth,'S0194',CGK.ZonePointers_s)
    shp=(self.context[CGK.NumberOfSteps_s][pth],)
    rs=val_u.checkChildValue(itv,shp,[CGK.I4, CGK.I8],pth,log,rs)
    rs=val_u.checkChildValue(tmv,shp,[CGK.R4, CGK.R8],pth,log,rs)
    rs=val_u.checkChildValue(znb,shp,[CGK.I4, CGK.I8],pth,log,rs)
    rs=val_u.checkChildValue(fnb,shp,[CGK.I4, CGK.I8],pth,log,rs)
    if (None not in [znb, zpt]):
      zmax=znb[1].max()
      if (CGU.getShape(zpt)!=(32,zmax,self.context[CGK.NumberOfSteps_s][pth])):
          rs=log.push(pth,'S0195',zpt[0])
      elif (CGU.getValueDataType(zpt) != CGK.C1):
          rs=log.push(pth,'S0196',zpt[0])
      else:
        d=zpt[1].T
        for dlist in d:
          for dname in dlist:
            tname=dname.tostring().rstrip()
            tnode=CGU.hasChildName(parent,tname)
            if ((tname!=CGK.Null_s) and (tnode is None)):
              rs=log.push(pth,'S0501',tname,zpt[0])
    if (None not in [fnb, fpt]):
      fmax=fnb[1].max()
      if (CGU.getShape(fpt)!=(32,fmax,self.context[CGK.NumberOfSteps_s][pth])):
          rs=log.push(pth,'S0195',fpt[0])
      elif (CGU.getValueDataType(fpt) != CGK.C1):
          rs=log.push(pth,'S0196',fpt[0])
      else:
        d=fpt[1].T
        for dlist in d:
          for dname in dlist:
            tname=dname.tostring().rstrip()
            tnode=CGU.hasChildName(parent,tname)
            if ((tname!=CGK.Null_s) and (tnode is None)):
              rs=log.push(pth,'S0501',tname,fpt[0])
    # TODO: check 503
    return rs
  # --------------------------------------------------------------------
  def ZoneIterativeData_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    bpath=CGU.getPathAncestor(pth,level=2)
    bnode=CGU.getNodeByPath(tree,bpath)
    bitnode=CGU.hasChildType(bnode,CGK.BaseIterativeData_ts)
    if (bitnode is None):
      rs=log.push(pth,'S0193',CGK.BaseIterativeData_ts,bpath)
    for ptrname in CGK.PointerNames_l:
      ptrnode=CGU.hasChildName(node,ptrname)
      if (ptrnode is not None):
        if (ptrnode[3]!=CGK.DataArray_ts):
          rs=log.push(pth,'S0003')
        elif (CGU.getValueDataType(ptrnode) not in [CGK.C1]):
          rs=log.push(pth,'S0004')
        elif (ptrnode[1].shape!=(32,self.context[CGK.NumberOfSteps_s][pth])):
          rs=log.push(pth,'S0192',ptrnode[1].shape,
                      (32,self.context[CGK.NumberOfSteps_s][pth]),ptrname)     
        d=ptrnode[1].T
        for dname in d:
          tname=dname.tostring().rstrip()
          tnode=CGU.hasChildName(parent,tname)
          if ((tname!=CGK.Null_s) and (tnode is None)):
            rs=log.push(pth,'S0501',tname,ptrname)
    return rs
  # --------------------------------------------------------------------
  def RigidGridMotion_t(self,pth,node,parent,tree,log):
    rs=CGM.CHECK_OK
    shp=(self.context[CGK.PhysicalDimension_s][pth],2)
    if ((node[1] is None)
        or (not CGU.stringValueInList(node,CGK.RigidGridMotionType_l))):
      rs=log.push(pth,'S0107')
    ol=CGU.hasChildName(node,CGK.OriginLocation_s)
    if (ol is None):
      rs=log.push(pth,'S0194',CGK.OriginLocation_s)
    else:
      rs=val_u.checkChildValue(ol,shp,[CGK.R4, CGK.R8],pth,log,rs)
    shp=(self.context[CGK.PhysicalDimension_s][pth],)
    rs=val_u.checkChildValue(CGU.hasChildName(node,CGK.RigidRotationRate_s),
                             shp,[CGK.R4, CGK.R8],pth,log,rs)
    rs=val_u.checkChildValue(CGU.hasChildName(node,CGK.RigidRotationAngle_s),
                             shp,[CGK.R4, CGK.R8],pth,log,rs)
    rs=val_u.checkChildValue(CGU.hasChildName(node,CGK.RigidVelocity_s),
                             shp,[CGK.R4, CGK.R8],pth,log,rs)
    return rs

# -----
