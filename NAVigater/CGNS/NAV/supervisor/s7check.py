#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from CGNS.pyCGNSconfig import version as __vid__
import CGNS.NAV.gui.s7globals
G___=CGNS.NAV.gui.s7globals.s7G

import CGNS.VAL.parse.PYparse as parser
import CGNS.NAV.gui.s7utils as s7u
import re
import imp
import sys

class s7DefaultLog:
  def __init__(self):
    self.buff=''
  def push(self,msg):
    self.buff+=msg

class s7Query:
  qName='Name'
  qType='Type'
  qValue='Value',
  qParentName='Parent Name'
  qParentType='Parent Type'
  qParentValue='Parent Value'
  qPath='Path'
  qDataType='DataType'
  taglist=[qName,qType,qValue,
           qParentName,qParentType,qParentValue,
           qPath,qDataType]
  OR='or'
  NOT='not'
  AND='and'
  conlist=[OR,NOT,AND]
  def __init__(self,Q,name=None,comment=None):
    self.Q=Q
    if (name!=None):    self.name=None
    if (comment!=None): self.comment=None
  def save(self,filename,date):
    f=open(filename,'w+')
    f.write("# pyS7 - Query definition - %s\n#\n"%date)
    f.write("name='%s'\ncomment=\"\"\"%s\"\"\"\n"%(self.name,self.comment))
    f.write("query=%s\n#\n"%self.Q)
    f.close()
  def load(self,fd,fn):
    sprev=sys.path
    sys.path.append(fd)
    try:
      m=imp.find_module(fn)
      if (s7u.getFileSize(m[1]) > G___.minFileSizeNoWarning):
        if (not s7u.bigFileWarning(G___.noData)): return None
      t=imp.load_module(fn,m[0],m[1],m[2])
      self.Q=t.query
      self.name=t.name
      self.comment=t.comment
    except ValueError: pass
    sys.path=sprev
  def evalQuery(self,tree,node,parentnode):
    return self.evalSubQuery(self.Q,tree,node,parentnode)
  def evalSubQuery(self,qry,tree,node,parentnode):
    if (type(qry)==type([])):
      operator=qry[0]
      querylst=qry[1:]
      if (operator==s7Query.AND):
        r=1
        for q in querylst:
          rn=self.evalSubQuery(q,tree,node,parentnode)
          r*=rn
          if (not r): return 0
        return 1
      if (operator==s7Query.OR):
        r=0
        for q in querylst:
          r=r+self.evalSubQuery(q,tree,node,parentnode)
          if (r): return 1
        return 0
      if (operator==s7Query.NOT):
        return not self.evalSubQuery(q,tree,node,parentnode)
    else:
      vartag=qry[0]
      nvalue=qry[1]
      match=1
      if (vartag==s7Query.qName):
        match=re.match(nvalue,node[0])
      if (vartag==s7Query.qType):
        match=re.match(nvalue,node[3])
      if (vartag==s7Query.qDataType):
        match=re.match(nvalue,s7parser.getNodeType(node))
      if (vartag==s7Query.qPath):
        match=re.match(nvalue,s7parser.getPathFromNode(tree,node))
      if (vartag==s7Query.qValue):
        pass
      if (vartag==s7Query.qParentName):
        pass
      if (vartag==s7Query.qParentType):
        pass
      if (vartag==s7Query.qParentValue):
        pass
      return (not (match == None))

import CGNS.VAL.grammars.SIDSbase

# --------------------------------------------------
def checkTree(path,node,parent,tree,check,log):
  parser=CGNS.VAL.grammars.SIDSbase.SIDSpython()
  return parser.checkTree(path,node,parent,tree,check,log)

# --------------------------------------------------
def getStatusForThisNode(pth,node,parent,tree):
  parser=CGNS.VAL.grammars.SIDSbase.SIDSpython()
  return parser.getStatusForThisNode(pth,node,parent,tree)

# --------------------------------------------------
def getEnumerateList(node):
  parser=CGNS.VAL.grammars.SIDSbase.SIDSpython()
  return parser.getEnumerateList(node)

# --------------------------------------------------  
def run(args):
  filename=sys.argv[1]
  import CGNS
  lk=CGNS.getLinksAsADF('%s/%s%s'%(fd,fn,fileext))
  tt=CGNS.loadAsADF('%s/%s%s'%(fd,fn,fileext),G___.followLinks,vmax)
  parent=None
  strlog=s7DefaultLog()
  for node in tree[2]:
    parser.checkTree('/'+node[0],node,parent,tree,1,strlog)

# --------------------------------------------------------------------
