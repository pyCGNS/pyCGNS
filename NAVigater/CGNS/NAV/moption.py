#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import shutil
import os
import os.path as OP
import sys
import imp
from time import time,gmtime,strftime

from CGNS.NAV.defaults import G__HISTORYFILENAME,G__OPTIONSFILENAME

def writeFile(tag,name,udict,filename):
  gdate=strftime("%Y-%m-%d %H:%M:%S", gmtime())
  s="""# CGNS.NAV - %s file - Generated %s\n%s={\n"""%(tag,gdate,name)
  for k in udict:
    val=str(udict[k])
    if (type(udict[k]) in [unicode, str]): val="'%s'"%str(udict[k])
    s+="""'%s':%s,\n"""%(k,val)
  s+="""} # --- last line\n"""
  f=open(filename,'w+')
  f.write(s)
  f.close()

def readFile(name,filename):
  dpath='/tmp/pyCGNS.tmp:%d.%s'%(os.getpid(),time())
  os.mkdir(dpath)
  try:
    shutil.copyfile(filename,dpath+'/%s.py'%name)
  except IOError:
    shutil.rmtree(dpath)    
    return None
  sprev=sys.path
  sys.path=[dpath]+sys.path
  try:
    return sys.modules[name]
  except KeyError:
    pass
  fp, pathname, description = imp.find_module(name)
  try:
    mod=imp.load_module(name, fp, pathname, description)
  finally:
    if fp:
       fp.close()
  shutil.rmtree(dpath)
  sys.path=sprev
  return mod

def trpath(path):
  return OP.normpath(OP.expanduser(OP.expandvars(path)))

def writeHistory(control):
  filename=trpath(G__HISTORYFILENAME)
  writeFile('History','history',control._history,filename)

def readHistory(control):
  filename=trpath(G__HISTORYFILENAME)
  m=readFile('history ',filename)
  if (m is None): return None
  return m.history

def writeOptions(control):
  filename=trpath(G__OPTIONSFILENAME)
  writeFile('User options','options',control._options,filename)
  
def readOptions(control):
  filename=trpath(G__OPTIONSFILENAME)
  m=readFile('options ',filename)
  if (m is None): return None
  return m._options

# -----------------------------------------------------------------
