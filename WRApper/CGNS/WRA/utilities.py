#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  -------------------------------------------------------------------------
import CGNS.WRA
import CGNS.WRA._mll     as MLL
import CGNS.WRA.wrapper  as WRP
import CGNS.WRA._adf     as ADF
import numpy             as NPY
import os
import string

def loadPythonFileAsImport(name,syspath=None):
  import imp
  fp, fn, fd = imp.find_module(name,syspath)
  md=imp.load_module(name,fp,fn,fd)
  return md

# ======================================================================
def zoneByName(db, name, base=1):
    for zi in range(1,db.nzones(1)+1):
      zn=db.zoneread(base,zi)[2]
      if (zn == name):
        return zi
    raise WRP.CGNS_NoSuchZone, name

# ======================================================================
def boundaryByName(db, zid, name, base=1):
    for bi in range(1,db.nbc(1,1)+1):
      bn=db.bcinfo(base,zid,bi)[0]
      if (bn == name):
        return bi
    raise WRP.CGNS_NoSuchBoundary, name

# ======================================================================
def flowSolutionByName(db, zi,  name):
    for si in range(1,db.nsols(1,zi)+1):
      sn=db.solinfo(1,zi,si)
      if (sn[1] == name):
        return si
    raise WRP.CGNS_NoSuchFlowSolution, name

# ======================================================================
def arrayByName(db, name):
    """Get an array by its name.
goto should have been performed before calling this function."""
    aname="No array found in this node"
    for ai in range(1,db.narrays()+1):
      aname=db.arrayinfo(ai)[0]
      if (aname == name):
        return ai
    raise WRP.CGNS_NoSuchArray, aname

# ======================================================================
def userDataByName(db, name):
    """Get a user data node by its name.
goto should have been performed before calling this function."""
    aname="No userdata found in this node"
    for ai in range(1,db.nuserdata()+1):
      aname=db.userdataread(ai)[1]
      if (aname == name):
        return ai
    raise WRP.CGNS_NoSuchArray, aname

# ======================================================================
def setBaseIterativeTime(db,index,value,base=1,baseiterativedata=1):
    """Set a time value in the base iterative array.
The array should have been written before.
We have base: 1 and baseiterativedata: 1 as befaults."""
    db.goto(base,[(MLL.BaseIterativeData_t,baseiterativedata)])
    ai=arrayByName(db,MLL.TimeValues)
    dim=db.arrayinfo(ai)[3]
    #print db.arrayinfo(ai)
    v=db.arrayread(ai)
    v[index]=value
    db.arraywrite(MLL.TimeValues,MLL.RealDouble,1,dim,v)    
  
# ======================================================================
def setZoneIterativePointer(db,zone,index,name,base=1,zoneiterativedata=1):
    """Set a flow solution name in the zone iterative data list.
The array should have been written before.
We have base: 1 and zoneiterativedata: 1 as defaults."""
    db.goto(base,[(MLL.Zone_t,zone),
                  (MLL.ZoneIterativeData_t,zoneiterativedata)])
    ai=arrayByName(db,MLL.FlowSolutionsPointers)
    dim=db.arrayinfo(ai)[3]
    #print db.arrayinfo(ai)
    v=db.arrayread(ai)
    v[index]="%-32.32s"%name
    db.arraywrite(MLL.FlowSolutionsPointers,MLL.Character,2,dim,v)

# ======================================================================
def getUserDefinedNode(db,name):
   n=db.nuserdata()
   for nn in range(1,n+1):
      nnn=db.userdataread(nn)
      if (nnn[1] == name): return nn
   return 0

# ======================================================================
def singleValue(db,name,value):
   db.arraywrite(name,MLL.RealDouble,1,(1,),NPY.array([value],'d'))

# ======================================================================
def reverse(dims):
   for n in len(dims):
       pass

# ======================================================================
def __pnice(a,t):
   if (type(a) == type(3.14)):
     return "%g"%a
   if (type(a) in [ type((1,)), type([]) ]):
     if (len(a) >= 1):
       if not t: # list/array
         r="["
       else:
         r="("
       for n in range(len(a)):
           r+="%s, "%__pnice(a[n],t)
       if (len(r)>1):
           r=r[:-2]
       if not t:
           r+="]"
       else:
           r+=")"
       return r
   else:
       return str(a)
 
def pnice(a):
   if (type(a) == type((1,))):
     return __pnice(a,1)
   else:
     return __pnice(a,0)  

# ======================================================================
def __singleNodeFromADF(db,id,path,flink,maxread,dmax,ptarget):
  nodeinfo=db.nodeAsDict(id)
  if (path and (len(path)==1) and (path[0] == nodeinfo['name'])):
    rinfo=__parseAndReadADF(db,id,flink,maxread,dmax,ptarget)
    return rinfo
  if (path and (path[0] == nodeinfo['name'])):
    clist=list(nodeinfo['children'])
    for child in clist:
      rinfo=__singleNodeFromADF(db,db.get_node_id(id,child),
                                path[1:],flink,maxread,dmax,ptarget)
      if (rinfo!=None): return rinfo
  return None
  
def getSingleNodeFromADF(file,path,flink,maxread,dmax,ptarget):
  db=WRP.pyADF(file,ADF.READ_ONLY,ADF.NATIVE)
  nodeinfo=db.nodeAsDict(db.root())
  for child in nodeinfo['children']:
    info=__singleNodeFromADF(db,db.get_node_id(db.root(),child),
                             string.split(path,'/')[1:],
                             flink,maxread,dmax,ptarget)
    if (info!=None): return info
  db.database_close()
  return info

# ======================================================================
def search(node,path):
  lpath=path.split('/')
  if (len(path) and (path[0]=='/')):
    lpath=lpath[1:]
  return searchNode(node,lpath)
  
def searchNode(node,lpath):
  if (node==None):    return None
  if (len(node)!=4):  return None
  if (node[2]==None): return None  
  for n in node[2]:
    if (n[0]==lpath[0]):
      if (len(lpath)>1):         
        return searchNode(n,lpath[1:])
      elif (len(lpath)==1):
        return n      
  return None
  
def searchNodeType(node,vtype):   
  result=[]
  return searchType(node,vtype,result)
  
def searchType(node,vtype,result):
  for n in node[2]:        
    if (n[3]==vtype):
      result.append(n)
    else :
      searchType(n,vtype,result)      
  return result

def searchNodeValue(node,value):   
  result=[]
  return searchValue(node,value,result)

def searchValue(node,value,result):   
  for n in node[2]:        
    if (n[1]==value):     
      result.append(n)
    else :
      searchType(n,value,result)      
  return result

# -----------------------------------------------------------------------------
def isArray(ar):
  return (type(ar) == type(NPY.ones((1,))))

def arraySize(ar,dt):
  if (dt != 'C1'):
    if (NPY.isfortran(ar)):
      sz=list(ar.shape)
      sz.reverse()
      return tuple(sz)
    return ar.shape
  else:
    return (len(ar.tostring()),)
  
def arrayValue(ar,dt):
  if (dt == 'C1'): ndt='c'
  if (dt == 'I4'): ndt='i'
  if (dt == 'R4'): ndt='f'
  if (dt == 'R8'): ndt='d'  
  if (isArray(ar)):
    if (ar.dtype.char == ndt): return ar
    else :                     return ar.astype(ndt)
  if (type(ar) == type("")):   return NPY.array([ar]).astype(ndt)
  if (type(ar) == type(4)):    return NPY.array([ar]).astype(ndt)
  if (type(ar) == type(4.2)):  return NPY.array([ar]).astype(ndt)
  if (type(ar) == type((4,))): return NPY.array([ar]).astype(ndt)
  if (type(ar) == type([4,])): return NPY.array([ar]).astype(ndt)
  else:
    return None
  
# -----------------------------------------------------------------------------
# duplicated with cgnshdf.py
def nodeType(node):
  data=node[1]
  if (node[0] == 'CGNSLibraryVersion_t'):
    return 'R4' # ONLY ONE R4 IN ALL SIDS !
  if ( data == None ):
    return 'MT'
  if (type(data) == type("")):
    return 'C1'
  if ( (type(data) == type(1.2))):
    return 'R8'
  if ( (type(data) == type(1))):
    return 'I4'
  if ( (type(data) == type(NPY.ones((1,)))) ):
    if (data.dtype.char in ['S','c']):        return 'C1'
    if (data.dtype.char in ['f','F']):        return 'R4'
    if (data.dtype.char in ['D','d']):        return 'R8'
    if (data.dtype.char in ['l','i','I']):    return 'I4'
  if ((type(data) in [type([]),type((1,))]) and (len(data))): # oups !
    if (type(data[0]) == type("")):           return 'C1' 
    if (type(data[0]) == type(0)):            return 'I4' 
    if (type(data[0]) == type(0.0)):          return 'R8'
  return '??'

# -----------------------------------------------------------------------------
def __parseAndWriteADF(db,tree,parent_id,links,path): 
  ar=None
  sz=None 
  isLink=0
  lpath=string.join(path+[tree[0]],'/')
  for lk in links:
    if (lk[1]==lpath):
      isLink=1
      break
  if (not isLink):
    id=db.create(parent_id,tree[0])        
    db.set_label(id,tree[3])
    ntype=nodeType(tree)
    value=tree[1]

    if (value != None):
      tvalue=arrayValue(value,ntype)
      sz=arraySize(tvalue,ntype)
      ar=tvalue
      db.put_dimension_information(id,ntype,sz)        
      db.write_all_data(id,ar) 
    if (tree[2]):
      for tchild in tree[2]:
        __parseAndWriteADF(db,tchild,id,links,path+[tree[0]])
  else:
    db.link(parent_id,tree[0],lk[2],lk[3])
  return db  
      
# -----------------------------------------------------------------------------
def __parseAndReadADF(db,nodeid,__followlink=1,__maxreadalldata=0,
                      __depthmax=999,__pathtarget=None,dbs={}):
  __depthmax-=1
  nodeinfo=db.nodeAsDict(nodeid)
  clist=list(nodeinfo['children'])
#  print nodeinfo['name']
  if (__pathtarget!=None):
    if (__pathtarget==[]): clist=[]
    else:
      rlist=[]
      for n in clist:
#        print 'loadAsADF ',n,__pathtarget[0]
        if (n == __pathtarget[0]):
          __pathtarget=__pathtarget[1:]
          rlist=[n]
          break
      clist=rlist
  if (nodeinfo['dimensions'] == ()): 
    ar=None     
  else:
    maxsize=reduce(lambda x,y: x*y,nodeinfo['dimensions']) 
    if ((not __maxreadalldata) or (nodeinfo['label']!='DataArray_t')):
      ar=db.read_all_data(nodeinfo['id'])
    # avoid read DataArray_t unless..
    # - it is C1
    # - max size < __maxreadalldata
    elif (((nodeinfo['label']=='DataArray_t') and (nodeinfo['datatype']=='C1'))
      or  ((nodeinfo['label']=='DataArray_t') and (maxsize<__maxreadalldata))):
      ar=db.read_all_data(nodeinfo['id'])
    else:
      ar=None
  #if ((ar!=None) and (nodeinfo['label']=='IndexRange_t')): print ar
  #link
  if ((nodeinfo['datatype']=='LK') and (__depthmax>0)):      
      linkFile=nodeinfo['file']
      linkNode=nodeinfo['path']
      if (__trustLink(linkFile,linkNode) and __followlink):
        actualinfo=getSingleNodeFromADF(linkFile,linkNode,
                                        __followlink,__maxreadalldata,
                                        __depthmax,__pathtarget)
        if (actualinfo==None):
          result=loadAsADF(linkFile,
                           __followlink,__maxreadalldata,
                           __depthmax,__pathtarget,dbs,0)
          info=search([None, None, result, None], linkNode)
          if (info==None):
            info=[nodeinfo['name'],
                  '%s:%s'%(linkFile,linkNode),[],nodeinfo['label']]
        info=[nodeinfo['name'],actualinfo[1],actualinfo[2],actualinfo[3]]
      else:
        info=[nodeinfo['name'],
              '%s:%s'%(linkFile,linkNode),[],nodeinfo['label']]
  #no link  
  elif (__depthmax>0): 
    info=[nodeinfo['name'],ar]     
    lc=[]
    for n in clist:
      lc.append(__parseAndReadADF(db,db.get_node_id(nodeid,n),
                                  __followlink,__maxreadalldata,
                                  __depthmax,__pathtarget,dbs))
    info+=[lc,nodeinfo['label']]
  else:
    info=[nodeinfo['name'],ar]     
    lc=[]
    info+=[lc,nodeinfo['label']]
  return info    

def __trustLink(file,path):
  tfile=file
  try:
    if (not os.path.exists(tfile)):
      print '## pyS7: lk error: no such file[%s]'%file
      return (0,file)
    return (1,tfile)
  except:
    return (0,file)

def __parseAndFindLinksADF(db,nodeid,level,path,dbs,file):
  nodeinfo=db.nodeAsDict(nodeid)
  clist=list(nodeinfo['children'])
  if (nodeinfo['name']not in ['ADF MotherNode','HDF5 MotherNode']):
    path+='/%s'%nodeinfo['name']
  #link
  if (nodeinfo['datatype']=='LK'):
    linkFile=nodeinfo['file']
    linkNode=nodeinfo['path']
    (tlk,tfile)=__trustLink(linkFile,linkNode)
    result=[(file,path,tfile,linkNode,level,tlk)]
    r=None
    if (tlk):
      #print '__parseAndFindLinksADF',level,tfile,linkNode
      r=findLinkAsADF(tfile,level+1,'',dbs)
    if (r): result+=r
  #no link  
  else:
    result=[]
    for n in clist:
      r=__parseAndFindLinksADF(db,db.get_node_id(nodeid,n),level,path,dbs,file)
      if (r): result+=r
  return result

# -----------------------------------------------------------------------------
def findLinkAsADF(file,level=0,path='',dbs={}):
#  print 'findLinkAsADF',file, level
#  print 'findLinkAsADF',dbs
  lastfileentry='  last  '
  __links = None
  rfile=os.path.split(file)[-1]
  dfile=os.path.split(file)[0]  
  if (rfile==''):
    rfile=lastfileentry
    for f in dbs:
      if ((dbs[f]==dbs[rfile]) and (f!=lastfileentry)): rfile=f
  try:
    if (not dbs.has_key(rfile)):
      #print 'pyCGNS: findLinkAsADF',file
      db=WRP.pyADF(file,ADF.READ_ONLY,ADF.NATIVE)
      dbs[rfile]=db
      dbs[lastfileentry]=db
      __links=__parseAndFindLinksADF(db,db.root(),level,path,dbs,file)
    if (level==0):
      kdbs=dbs.keys()
      for kdb in kdbs:
        dbs[kdb].database_close()
        del dbs[kdb]
  except ADF.error,e:
      print 'findLinkAsADF :',e
  return __links

# remove duplicates
def getLinksAsADF(file):
  r=findLinkAsADF(file,dbs={})
  if (not r): return []
  r.sort()
  s={}
  for i in r:
    s[str(i)]=i
  r=[]
  for i in s:
    r.append(s[i])
  return r
       
# -----------------------------------------------------------------------------
# load an ADF/HDF file
#
# if max != 0: max is the max size of DataArray_t to load,
#              else None is returned if size>max
# if max == 0: always return DataArray_t whichever size they have
#
# depth : stop recursion when node depth is reached (or no more child)
#
# if path != None: go straight to given path, other children are ignored
# if path == None: parse children regardless of their name
#
def loadAsADF(file,link=1,max=0,depth=999,path=None,dbs={},start=1):
  __tree = None
  lastfileentry='  last  '
  rfile=os.path.split(file)[-1]
  dfile=os.path.split(file)[0]  
  if (rfile==''):
    rfile=lastfileentry
    for f in dbs:
      if ((dbs[f]==dbs[rfile]) and (f!=lastfileentry)): rfile=f
  try:     
    if (not dbs.has_key(rfile)):
      #print 'pyCGNS: loadAsADF (open) ',file,path
      db=WRP.pyADF(file,ADF.READ_ONLY,ADF.NATIVE)
      r=__parseAndReadADF(db,db.root(),link,max,depth,path,dbs)
      db.database_close()
      dbs[rfile]=r[2]
      dbs[lastfileentry]=r[2]
      __tree=r[2]
    else:
      #print 'pyCGNS: loadAsADF (cache)',file,path
      __tree=dbs[rfile]
    if (start):
      #print 'pyCGNS: loadAsADF free cache'
      kdbs=dbs.keys()
      for kdb in kdbs:
        del dbs[kdb]
  except ADF.error,e:
      print e
  return __tree

# -----------------------------------------------------------------------------
def saveAsADF(file,tree=None,links=[]): 
  import os
  bExist = os.path.isfile(file)
  try:   
    # check tree types here
    db=WRP.pyADF(file,ADF.NEW,ADF.NATIVE)
    path=[''] 
    for tc in tree[2]:
      __parseAndWriteADF(db,tc,db.root(),links,path) 
    db.database_close()
  except KeyError: #CGNS.adf.error,e:
    pass #print e

# -----------------------------------------------------------------------------
def savepy(filePy,data):  
  """-save memory tree 
  ''*savepy'(filepy:S,data:A)'  
   data : memory tree
   filepy : path file.py   
  """ 

  f=open(filePy,"w+")
  f.write("from numpy import *\ndata=")
  f.write(str(data))
  f.close()


# -----------------------------------------------------------------------------
def loadpy(filepy,path):

  """-load tree py format-
  'data:N='*loadpy'(filepy:S,path:[*S])'  
   data is memory tree  """ 
  import imp 
  import sys

  try:
    return sys.modules["name"]
  except KeyError:
    pass
  fp,chemin,description=imp.find_module(filepy,path)
  try:   
    pyf = imp.load_module(filepy,fp,chemin,description)
    return pyf.data
  finally:
    if fp:
      fp.close()

# -----------------------------------------------------------------------------
def prettystringone(name,value,ntype,subtree,path,count):
  sreturn=""
  path+="%s"%name
  if   (type(value) == type("")): vp= "'%s'"%value
  else:
      try:
          value.shape
          if (value.typecode() == NPY.UInt8): # strings are UInt8
            vp='"""%s"""'%value.tostring()
          else:    
            vp="NPY.%s"%repr(value)
      except AttributeError:
          vp=value
  sreturn+=count*" "+"['%s', %s,"%(name,vp)
  if (subtree == []):  sreturn+="[],"
  else:                sreturn+="[\n"
  for n in subtree:
      sreturn+=prettystringone(n[0],n[1],n[3],n[2],path+"/",count+1)
  sreturn+=count*" "
  if (subtree != []):  sreturn+="],"
  sreturn+="'%s'],\n"%(ntype)
  return sreturn

# -----------------------------------------------------------------------------
def prettystring(tree):
  s=""
  if (isRootNode(tree)): # skip CGNSTree bad-node
    s+="[\n"
    for nodes in tree[2]:
      checkNode(nodes,dienow=1)
      node=nodes
      path="/"
      s+=prettystringone(node[0],node[1],node[3],node[2],path,1)
    s+="]\n"        
  else:
    checkNode(tree,dienow=1)
    node=tree
    path="./"
    s+=prettystringone(node[0],node[1],node[3],node[2],path,0)
  return s
    
# last line
