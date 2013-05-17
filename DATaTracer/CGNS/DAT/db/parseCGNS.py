#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
# pyDAX - DBMS schema - Parse CGNS file
# ------------------------------------------------------------

#
from   string                   import *
import CGNS.DAT.db.adfutils     as dxUT
import CGNS.DAT.db.foreignKeys  as dxFK
from   CGNS.DAT.utils           import *
from   CGNS.DAT.exceptions      import *
import CGNS.DAT.utils           as ut
import CGNS
import numpy                    as NPY
# ----------------------------------------------------------------------
ddefault=NPY.array(tuple("00/00/0000 00:00:00"),dtype='S1')
sdefault=NPY.array(tuple("<NO VALUE>"),dtype='S1')
idefault=NPY.array(-1,NPY.int32)
fdefault=NPY.array(-1.,NPY.float64)
#
# --- what should be found as Meta data, default values if it has to be created
basetitle       = '.MetaData/Title'
basedescription = '.MetaData/Description'
baseremarks     = '.MetaData/Remarks'
owner           = '.MetaData/Owner'
version         = '.MetaData/Version'
release         = '.MetaData/Release' 
change          = '.MetaData/Change'
creationdate    = '.MetaData/CreationDate'
modificationdate= '.MetaData/ModificationDate'
policy          = '.MetaData/Policy'
status          = '.MetaData/Status'
#
# --- CGNS/Dax interface storage
cgnsfilename    = 'cgnsfilename'
basename        = 'basename'
nzones          = 'nzones'
nlinks          = 'nlinks'
linkslist       = 'linkslist'
pdim            = 'pdim'
cdim            = 'cdim'
#
# --- aliases
simulation      = 'SimulationType'
flowequation    = 'FlowEquationSet'
#
# --- Attributes
refdct={
'.MetaData/CheckSum':         sdefault,
'.MetaData/GlobalCheckSum':   sdefault,
creationdate            :     ddefault,
modificationdate            : ddefault,
'.MetaData/Platform':         sdefault,
'.MetaData/Memory':           sdefault,
'.MetaData/Time':             sdefault,
owner:                        NPY.array(tuple('X'),dtype='S1'),
policy:                       NPY.array(tuple('NONE'),dtype='S1'),
status:                       NPY.array(tuple('DRAFT'),dtype='S1'),
version:                      idefault,
release:                      idefault,
change:                       idefault,
basetitle:                    sdefault,
basedescription:              sdefault,
baseremarks:                  sdefault,
'.CHANCE/CaseNumber':         sdefault,
'.CHANCE/Geometry':           sdefault,
'.CHANCE/Family':             sdefault,
'.CHANCE/Measurements':       sdefault,
'.CHANCE/Remarks':            sdefault,
}
# False below means no mapping to "open" attribute list, a switch case
# will be found in the update method. 
#
withinDAXModifiableAttributes={
 owner:           False,
 version:         False,
 release:         False,
 change:          False,
 policy:          False,
 status:          False,
 basetitle:       False,
 basedescription: True,
 baseremarks:     True
}
# ----------------------------------------------------------------------
defaultDict={
 owner:'<NOVALUE>',
 pdim:0,
 cdim:0,
 simulation:'<NOVALUE>',
}
# ----------------------------------------------------------------------
# USE APP instead 2010
class fileCGNS:
  def __init__(self,filename=None):
    if (filename):
      f=open(filename,'rb')
      self._binbuffer=f.read()
      f.close()
      self._adf=CGNS.pyADF(filename,CGNS.READ_ONLY,CGNS.NATIVE)
      self._name=filename
    else:
      self._name=None
      self._adf=None
      self._binbuffer='\0'
  def asBinaryString(self):
    return self._binbuffer
  def checksum(self):
    ck=0
    if (self._adf != None):
      ck=checksum(self._name,self._adf)
    return ck
  def open(self,filename,mode=0):
    self._name=filename
    f=open(filename,'rb')
    self._binbuffer=f.read()
    f.close()
    if (mode == 0):
      self._adf=CGNS.pyADF(filename,CGNS.READ_ONLY,CGNS.NATIVE)
    if (mode == 1):
      self._adf=CGNS.pyADF(filename,CGNS.OLD,CGNS.NATIVE)
  def __del__(self):
    if (self._adf): self._adf.database_close()
  def checkStructure(self):
    if (self._adf): checkMeta(self._adf,leave=1)
  def parse(self):
    if (self._adf): dct=parseCGNSfile(self._name,self._adf)
    return dct
  def update(self,dct):
    vdct={}
    for k in dct:
      vdct[k]=ut.getAsArray(dct[k])
    if (self._adf): changeMeta(self._adf,vdct)
# ----------------------------------------------------------------------
# If leave is not set, then we create the missing nodes...
#
def checkMeta(a,leave=0):
  b=dxUT.findChild(a,a.root(),CGNS.CGNSBase_t)[0]
  u=dxUT.findChild(a,b,CGNS.UserDefinedData_t)
  topnodesfound=[]
  for k in refdct:
    k1,k2=k.split('/')
    found1=0
    #print topnodesfound
    if (k1 not in topnodesfound):
      for ui in u:
        nodeinfo=a.nodeAsDict(ui)
        if (k1 == nodeinfo['name']):
          found1=1
          topnodesfound+=[k1]
          break
      if (not found1 and leave):
        raise DAXIncorrectCGNSFile("Top Node '%s' not found"%k1)
      if (not found1):
         nid=a.create(b,k1)
         a.set_label(nid,CGNS.UserDefinedData_t)
         #a.put_dimension_information(nid,'MT',(1,))
         topnodesfound+=[k1]
  for k in refdct:
    k1,k2=k.split('/')
    found2=0    
    u=dxUT.findChild(a,b,CGNS.UserDefinedData_t)
    for ui in u:
      nodeinfo=a.nodeAsDict(ui)
      if (k1 == nodeinfo['name']):
        kid=nodeinfo['id']
        m=dxUT.findChild(a,kid,CGNS.DataArray_t)
        for mi in m:
          cnodeinfo=a.nodeAsDict(mi)
          if (k2 == cnodeinfo['name']):
            found2=1
            break
        if (not found2 and leave):
          raise DAXIncorrectCGNSFile("Node '%s' not found"%k)
        if (not found2):
          nid=a.create(kid,k2)
          a.set_label(nid,CGNS.DataArray_t)
          dt=dxUT.getType(refdct[k])
          a.put_dimension_information(nid,dt,(len(refdct[k]),))
          a.write_all_data(nid,refdct[k])
# ----------------------------------------------------------------------
def changeMeta(a,dct):
  b=dxUT.findChild(a,a.root(),CGNS.CGNSBase_t)[0]
  u=dxUT.findChild(a,b,CGNS.UserDefinedData_t)
  for k in dct:
    k1,k2=k.split('/')
    found2=0    
    for ui in u:
      nodeinfo=a.nodeAsDict(ui)
      if (k1 == nodeinfo['name']):
        kid=nodeinfo['id']
        m=dxUT.findChild(a,kid,CGNS.DataArray_t)
        for mi in m:
          cnodeinfo=a.nodeAsDict(mi)
          if (k2 == cnodeinfo['name']):
            found2=1
            break
        if (found2):
          nid=cnodeinfo['id']
          dt=dxUT.getType(dct[k])
          a.put_dimension_information(nid,dt,(len(dct[k]),))
          a.write_all_data(nid,dct[k])
# ----------------------------------------------------------------------
NoAttributeNodeList=['ConvergenceHistory_t']
def getTheValue(lab,typ,dim,a,id):
  if (lab in NoAttributeNodeList): return ''
  if (len(dim) != 1):              return ''
  v=a.read_all_data(id)
  if (typ == 'C1'):    return v.tostring()
  if (typ == 'I4'):    return str(v.tolist())
  if (typ == 'R4'):    return str(v.tolist())
  if (typ == 'R8'):    return str(v.tolist())
# ----------------------------------------------------------------------
def findAllAttributes(fid,csr,a,id,currentpath,dct):
  node=a.get_name(id)
  currentpath=currentpath+"/"+node
  if a.is_link(id):
    dct[nlinks]+=1
    fn,nn=a.get_link_path(id)
    dct[linkslist].append([currentpath,fn,nn])
  else:
    nl=a.get_label(id)
    if (nl == CGNS.Zone_t):
      dct[nzones]+=1
    dt=a.get_data_type(id)
    dm=a.get_dimension_values(id)
    nv=getTheValue(nl,dt,dm,a,id)
    tp=((fid,nv,nl,currentpath,node),)
    keypath=join(currentpath.split('/')[2:],'/')
    dct[keypath]=nv
    for childname in a.children_names(id):
      cid=a.get_node_id(id,childname)
      rt=findAllAttributes(fid,csr,a,cid,currentpath,dct)
# ----------------------------------------------------------------------
def checkAttributes(attlist):
  for a in attlist:
    if (a not in withinDAXModifiableAttributes.keys()):
      raise DAXNoSuchAttribute("No update allowed on '%s'"%a)
# ----------------------------------------------------------------------
def parseCGNSfile(filename,adf=None):
  dct={}
  dct[cgnsfilename]=filename
  dct[nzones]=0
  dct[nlinks]=0
  dct[linkslist]=[]
  dct[simulation]="<NO VALUE>"
  dct[flowequation]=flowequation # watchdog for attribute table (not empty)
  if (not filename):
    dct[pdim]=0
    dct[cdim]=0
    for k in refdct:
      dct[k]=refdct[k]
    return dct
  #
  # get version
  if (adf):
    a=adf
  else:
    a=CGNS.pyADF(filename,CGNS.READ_ONLY,CGNS.NATIVE)
  id=a.get_node_id(a.root(),"CGNSLibraryVersion")
  vs=a.read_all_data(id)
  # parse every base
  for c in a.children_names(a.root()):
    cid=a.get_node_id(a.root(),c)
    lb=a.get_label(cid)
    if (lb == CGNS.CGNSBase_t):
        dct[basename]=a.get_name(cid)
        vs=a.read_all_data(cid)
        dct[pdim]=vs[1]       
        dct[cdim]=vs[0]
        findAllAttributes(filename,None,a,cid,"",dct)
  if (not adf): a.database_close()
  return dct
# ----------------------------------------------------------------------
def checkConsistency(dct,previousdct=None):
  if (xn(dct[cgnsfilename]) != dct[basename]):
    raise DAXIncorrectCGNSFile("Base name (%s) is not file ID (%s)"%
                               (dct[basename],xn(dct[cgnsfilename])))
  try:
    v=transAsPossible(dct[version])
    r=transAsPossible(dct[release])
    c=transAsPossible(dct[change])
    t=(v,r,c)
  except:
    DAXIncorrectCGNSFile("Version identifier incorrect (cannot read it)")
  if (   v not in range(0,100)
      or r not in range(0,100)
      or c not in range(0,100)):
    raise DAXIncorrectCGNSFile("Version identifier 'v%s.%s.%s' incorrect"%t)
#
