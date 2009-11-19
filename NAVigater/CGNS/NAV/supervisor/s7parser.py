#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.NAV.gui.s7globals
G___=CGNS.NAV.gui.s7globals.s7G

import CGNS.NAV.supervisor.s7cgnsTypes as CT
import CGNS.PAT.cgnskeywords
import s7grammar
import string
import numpy as Num
import re

__checkready=1
try:
  import CGNS.PAT.cgnskeywords as CK
except ImportError:
  __checkready=0

__keywordlist=CGNS.PAT.cgnskeywords.names

# --------------------------------------------------
def getNodeType(node):
  data=node[1]
  if (node[0] == 'CGNSLibraryVersion_t'):
    return 'R4' # ONLY ONE R4 IN ALL SIDS !
  if ( data in [None, []] ):
    return 'MT'
  if ( (type(data) == type("")) and (len(data)<G___.maxDisplaySize) ):
    return 'C1'
  if ( (type(data) == type(1.2))):
    return 'R8'
  if ( (type(data) == type(1))):
    return 'I4'
  if ( (type(data) == type(Num.ones((1,)))) ):
    if (data.dtype.char in ['S','c']):        return 'C1'
    if (data.dtype.char in ['f','F']):        return 'R4'
    if (data.dtype.char in ['D','d']):        return 'R8'
    if (data.dtype.char in ['l','i','I']):    return 'I4'
  if ((type(data) == type([])) and (len(data))): # oups !
    if (type(data[0]) == type("")):           return 'C1' 
    if (type(data[0]) == type(0)):            return 'I4' 
    if (type(data[0]) == type(0.0)):          return 'R8'
  print 'pyS7 debug: s7parser.getNodeType cannot find type of:', data
  return '??'

# --------------------------------------------------
def removeNodeFromPath(path,node):
  target=getNodeFromPath(path,node)
  if (len(path)>1):
    father=getNodeFromPath(path[:-1],node)
    father[2].remove(target)
  else:
    # Root node child
    for c in node[2]:
      if (c[0] == path[0]): node[2].remove(target)
    
# --------------------------------------------------
def getNodeFromPath(path,node):
  for c in node[2]:
    if (c[0] == path[0]):
      if (len(path) == 1): return c
      return getNodeFromPath(path[1:],c)
  return -1

# --------------------------------------------------
def getPathFromNode(node,rootnode,path="/"):
  if (node == rootnode):
    return path
  for c in rootnode[2]:
    if (path != '/'): path="%s/%s"%(path,c[0])
    else:             path="/%s"%(c[0])
    if (c == node):   return path
    else:             return getPathFromNode(tnode,c,path)
  return ''

# --------------------------------------------------
def childNames(node):
  r=[]
  if (node == None): return r
  for c in node[2]:
    r.append(c[0])
  return r

# --------------------------------------------------
def getNodeAllowedChildrenTypes(node):
  tlist=[]
  try:
    tlist+=CT.types[node[3]][0]
  except:
    pass
  return tlist

# --------------------------------------------------
def hasChildNodeOfType(node,ntype):
  if (node == None): return 0
  for cn in node[2]:
    if (cn[3]==ntype): return 1
  return 0

# --------------------------------------------------
def getTypeDataTypes(node):
  tlist=[]
  try:
    tlist+=CT.types[node[3]][1][0]
  except:
    pass
  return tlist

# --------------------------------------------------
def stringValueMatches(node,reval):
  if (node == None):            return 0
  if (node[1] == None):         return 0  
  if (getNodeType(node)!='C1'): return 0
  tn=type(node[1])
  if   (tn==type('')): vn=node[1]
  elif (tn == type(Num.ones((1,))) and (node[1].dtype.char in ['S','c'])):
    vn=node[1].tostring()
  else: return 0
  return re.match(reval,vn)

# --------------------------------------------------
def shift(path):
  n=string.split(path,'/')
  return len(n)*G___.shiftstring

# --------------------------------------------------------------------
def checkDataAndType(path,node,parent,log):
  dt=getNodeType(node)
  xt=getTypeDataTypes(node)
  if (dt not in xt):
    tag='#FAIL'
    clevel=2
    msg="%sBad data type [%s] for [%s] (expected one of %s)\n"%\
         (shift(path),dt,node[0],xt)
    log.push(msg,tag)
  pt=getNodeAllowedChildrenTypes(parent)
  if (pt==[]):
    pt=['CGNSBase_t','CGNSLibraryVersion_t']
  if (node[3] not in pt):
    tag='#FAIL'
    clevel=max(clevel,2)
    msg="%sBad node type [%s] for [%s]\n"%\
         (shift(path),node[3],node[0])
    log.push(msg,tag)
#  else:
#    tag=None
#    clevel=max(clevel,0)
#    msg="%sData type [%s] ok for [%s]\n"%(shift(path),node[3],node[0])
    
# --------------------------------------------------------------------
def checkName(path,node,parent,log):
  r=[]
  nm=node[0]
  tag=None
  clevel=0
  shft=shift(path)
  if (nm==''):
    tag='#FAIL'
    clevel=2
    log.push("%sName is empty string !\n"%shft,tag)
  for c in ['/','\\']:
    if (c in nm):
      tag='#FAIL'
      clevel=2
      log.push("%sForbidden char '%s' in name\n"%(shft,c),tag)
  for c in ['.','>','<','`',"'",'"']:
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
    
  cnlist=childNames(parent)
  if (cnlist):
    cnlist.remove(node[0])
    if (node[0] in cnlist):
      tag='#FAIL'
      clevel=2
      log.push("%sDuplicated node name [%s]\n"%(shft,node[0]),tag)

  return clevel
  
  
# --------------------------------------------------------------------
def checkLeaf(pth,node,parent,tree,check=1,log=None):
  if (log == None):      return 0
  if (not __checkready): return 0
  clevel=0
  clevel=max(checkName(pth,node,parent,log),clevel)
  clevel=max(checkDataAndType(pth,node,parent,log),clevel)
  try:
    ntype=node[3]
    if (ntype == 'int[1+...+IndexDimension]'):ntype="IndexRangeT1_t"
    if (ntype == 'int[IndexDimension]'):      ntype="IndexRangeT2_t"
    return apply(s7grammar.__dict__[ntype],[pth,node,parent,tree,check,log])
  except KeyError:
    return clevel
  
# --------------------------------------------------------------------
def checkTree(pth,node,parent,tree,check=1,log=None):
  log.push("\n%s\n"%(pth),'#INFO')
  clevel=checkLeaf(pth,node,parent,tree,check,log)
  r=[[pth,clevel]]
  parent=node
  for n in node[2]:
    r+=checkTree('%s/%s'%(pth,n[0]),n,parent,tree,check,log)
  return r
  
# --------------------------------------------------------------------
def getEnumerateList(node):
  try:
    ntype=node[3]+'_enum'
    return apply(s7grammar.__dict__[ntype],[])
  except KeyError:
    return None

# --------------------------------------------------------------------
def statusLeaf(pth,node,parent,tree):
  if (not __checkready): return 0
  return checkLeaf(pth,node,parent,tree,0)
  
# --------------------------------------------------------------------
def getStatusForThisNode(pth,node,parent,tree):
  stat=statusLeaf(pth,node,parent,tree)
  lpth=pth.split('/')
  if (lpth[0]==''):
    absolute=1
    if (len(lpth)>1): lpth=lpth[1:]
    else:             lpth=[]
  else:
    absolute=0
  if (node[0] in __keywordlist): return (G___.SIDSkeyword,stat, absolute)
  return (None,stat,absolute)

# --- last line
