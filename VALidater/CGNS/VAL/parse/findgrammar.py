#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import sys
import os
import fnmatch
import imp

PROFILENAME='grammars'

#  -------------------------------------------------------------------------
def readProfile():
  try:
    hdir=os.environ['HOME']
  except:
    return {}
  pdir='%s/.CGNS.NAV'%hdir
  if (not os.exists(pdir)):
    return {}
  sys.path.append(pdir)
  fp, pth, des = imp.find_module('grammars')
  try:
    mod=imp.load_module('grammars', fp, pth, des)
  finally:
    if fp:
      fp.close()
  return mod.Grammars
  
#  -------------------------------------------------------------------------
def findAllUserGrammars(verbose=False):
  kdict={}
  for pth in [p for p in sys.path if p!='']:
    if (verbose): print '### CGNS.VAL: scanning',pth
    try:
      for pthroot, dirs, files in os.walk(pth):
          for fn in files:
              if (fnmatch.fnmatch(fn,'CGNS_VAL_USER_*.py')
                  or fnmatch.fnmatch(fn,'CGNS_VAL_USER_*.so')):
                gkey=fn[14:-3]
                if (gkey in kdict):
                  if (verbose):
                    print '### CGNS.VAL:          found grammar:',
                    print gkey,'already found, ignore this one'
                else:
                  if (verbose):
                    print '### CGNS.VAL:          found grammar:',gkey
                  kdict[fn[14:-3]]=pthroot
    except OSError: pass
  return kdict

#  -------------------------------------------------------------------------
def findOneUserGrammar(tag,verbose=False):
  kdict={}
  found=False
  for pth in sys.path:
    if (verbose): print '### CGNS.VAL: scanning',pth
    try:
      for pthroot, dirs, files in os.walk(pth):
          for fn in files:
              if fnmatch.fnmatch(fn,'CGNS_VAL_USER_%s.py'%tag):
                  kdict[fn[14:-3]]=pthroot
                  found=True
                  break
          if found: break
    except OSError: pass
    if found: break
  return kdict

#  -------------------------------------------------------------------------
def importUserGrammars(key,recurse=False,verbose=False):
  mod=None
  modname='CGNS_VAL_USER_%s'%key
  try:
    tp=imp.find_module(modname)
  except ImportError:
    print '### CGNS.VAL [warning]: grammar [%s] not found\n'%key
    if (recurse):
      dk=findOneUserGrammar(key)
      if (key in dk):
          sys.path.append(dk[key])
          print '### CGNS.VAL [warning]: not in search path [%s]\n'%dk[key]
          try:
              tp=imp.find_module(modname)
          except ImportError:
              return None
      else:
          return None
    else:
      return None
  try:
    fp=tp[0]
    if (tp[2][2]!=imp.C_EXTENSION):
      mod=imp.load_module(modname, *tp)
    else:
      mod=imp.load_dynamic(modname,tp[1],tp[0])      
    # print '### CGNS.VAL [info]: Module info',tp
  finally:
    if fp:
       fp.close()
  return mod

# ---
