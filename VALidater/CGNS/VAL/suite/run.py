#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
#
import CGNS.VAL.simplecheck as CGV
import importlib
import sys
import os

def runSuite(suite,trace):
  s=importlib.import_module('.%s'%suite,'CGNS.VAL.suite')
  tlist=[]
  if (s is not None):
      p=os.path.split(s.__file__)[0]
      l=os.listdir(p)
      for t in l:
          if ((t[0]!='_') and (t[-3:]=='.py')): tlist.append(t[:-3])
  for t in tlist:
      tdlist=loadTree(suite,t)
      for (tag,T,diag) in tdlist:
          valTree(suite,t,tag,T,diag,trace)

def loadTree(key,test):
  m=importlib.import_module('.%s'%test,'CGNS.VAL.suite.%s'%key)
  return (m.TESTS)

def valTree(suite,t,tag,T,diag,trace):
    count=0
    r=CGV.compliant(T,False,['SAMPLE'])
    if (r[0]==diag): (v1,v2)=('pass','expected')
    else:            (v1,v2)=('FAIL','UNEXPECTED')
    if (trace): print '###'
    print '### [%s] %s/%s: %s'%(v1,suite,t,tag)
    if (trace):
      for m in r[1]:
        print '### %.2d %s error on: %s'%(count,v2,m[0])
        print '### %.2d %s'%(count,m[1])
        count+=1

tlist=('treebasics',)
for s in tlist:
    runSuite(s,True)

