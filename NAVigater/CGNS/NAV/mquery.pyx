#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import sys
import numpy

from PySide.QtCore    import *
from PySide.QtGui     import *

from CGNS.NAV.moption import Q7OptionContext as OCTXT
import CGNS.NAV.moption as OCST

import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK

# -----------------------------------------------------------------
def sameVal(n,v):
    if (n is None):
        if (v is None): return True
        return False
    if (n.dtype.char in ['S','c']): return (n.tostring() == v)
    if (n.dtype.char in ['d','f','i','l']): return (n.flat[0] == v)
    if (n==v): return True
    return False
# -----------------------------------------------------------------
def sameValType(n,v):
    if (type(n)!=numpy.ndarray): return False
    if (n.dtype.char==v): return True
    return False
# -----------------------------------------------------------------
def evalScript(node,parent,tree,path,val,args):
    l=locals()
    l[OCST.Q_VAR_RESULT_LIST]=[False]
    l[OCST.Q_VAR_NAME]=node[0]
    l[OCST.Q_VAR_VALUE]=node[1]
    l[OCST.Q_VAR_CGNSTYPE]=node[3]
    l[OCST.Q_VAR_CHILDREN]=node[2]
    l[OCST.Q_VAR_TREE]=tree
    l[OCST.Q_VAR_PATH]=path
    l[OCST.Q_VAR_USER]=args
    l[OCST.Q_VAR_NODE]=node
    pre=OCST.Q_SCRIPT_PRE+val+OCST.Q_SCRIPT_POST
    try:
      eval(compile(pre,'<string>','exec'),globals(),l)
    except Exception:
      RESULT=False
    RESULT=l[OCST.Q_VAR_RESULT_LIST][0]
    return RESULT
# -----------------------------------------------------------------
def parseAndSelect(tree,node,parent,path,script,args,result):
    path=path+'/'+node[0]
    Q=evalScript(node,parent,tree,path,script,args)
    R=[]
    if (Q):
        if (result):
            R=[Q]
        else:
            R=[path]
    for C in node[2]:
        R+=parseAndSelect(tree,C,node,path,script,args,result)
    return R

# -----------------------------------------------------------------
class Q7QueryEntry(object):
    def __init__(self,name,group=None,script=''):
        self._name=name
        self._group=group
        self._script=script
    @property
    def name(self):
        return self._name
    @property
    def group(self):
        return self._group
    @property
    def script(self):
        return self._script
    def setScript(self,value):
        self._script=value
    def __str__(self):
        s='("%s","%s","%s")'%(self.name,self.group,self._script)
        return s
    def run(self,tree,mode,*args):
        self._args=args
        result=parseAndSelect(tree,tree,[None,None,[],None],'',
                              self._script,self._args,mode)
        return result
    
# -----------------------------------------------------------------
