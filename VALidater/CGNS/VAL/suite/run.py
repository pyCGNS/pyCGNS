#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
#
import CGNS.VAL.simplecheck as CGV
import CGNS.MAP
import importlib
import sys
import os

KEY='SAMPLE'
FILES=False
CGNSCHECK=False

try:
  import CGNS.PRO
  KEY='SIDS'
  KEY='elsA'
except ImportError:
  pass

try:
  if (os.environ.has_key('CGNS_VAL_SAVE_FILES')):   FILES=True
except:
  pass

try:
  if (os.environ.has_key('CGNS_VAL_RUN_CGNSCHECK')): CGNSCHECK=True
except:
  pass

def cgcheck(filenametag):
  os.system('cgnscheck %s.hdf 1>%s.cgnscheck 2>&1'%(filenametag,filenametag))
  
def runSuite(suite,trace,savefile=False,cgnscheck=False):
  s=importlib.import_module('.%s'%suite,'CGNS.VAL.suite')
  tlist=[]
  count=1
  if (s is not None):
      p=os.path.split(s.__file__)[0]
      l=os.listdir(p)
      for t in l:
          if ((t[0]!='_') and (t[-3:]=='.py')): tlist.append(t[:-3])
  for t in tlist:
      tdlist=loadTree(suite,t)
      for (tag,T,diag) in tdlist:
          sr=valTree(suite,t,tag,T,diag,trace,count)
          if (diag): k='pass'
          else: k='fail'
          if (savefile):
            fname='%.4d-%s-%s-%s.hdf'%(count,k,suite,t)
            try:
              CGNS.MAP.save(fname,T)
            except CGNS.MAP.error,e:
              sr+='### * CHLone save error %d:%s\n'%(e[0],e[1])
            f=open('%.4d-%s-%s-%s.diag'%(count,k,suite,t),'w+')
            f.write(sr)
            f.close()
            if (cgnscheck):
              cgcheck('%.4d-%s-%s-%s'%(count,k,suite,t))
          count+=1

def loadTree(key,test):
  m=importlib.import_module('.%s'%test,'CGNS.VAL.suite.%s'%key)
  return (m.TESTS)

def valTree(suite,t,tag,T,diag,trace,count):
    r=CGV.compliant(T,False,[KEY])
    sr=''
    if (diag): k='pass'
    else: k='fail'
    if (r[0]==diag): (v1,v2)=('pass','expected')
    else:            (v1,v2)=('FAIL','UNEXPECTED')
    if (trace): sr+='###\n'
    sr='### [%s] %.4d-%s-%s-%s (%s)\n'%(v1,count,k,suite,t,tag)
    for m in r[1]:
      sr+='### > %s error on: %s\n'%(v2,m[0])
      sr+='### > %s\n'%(m[1])
    if (trace):
      print sr
    return sr

tlist=('treebasics','Onera')
for s in tlist:
    runSuite(s,True,FILES,CGNSCHECK)

