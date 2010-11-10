# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS PATterns
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
import os
import sys
import shutil
import re
from   distutils.dir_util import remove_tree

from  distutils.core import setup
from  distutils.util import get_platform
from  distutils.command.clean import clean as _clean

rootfiles=['errors.py']
compfiles=['__init__.py','midlevel.py','wrap.py','version.py']

# --------------------------------------------------------------------
def prodtag():
  from time import gmtime, strftime
  proddate=strftime("%Y-%m-%d %H:%M:%S", gmtime())
  prodhost=os.uname()
  return (proddate,prodhost)
  
# --------------------------------------------------------------------
def search(tag,deps=[]):
  import sys
  import distutils.util
  import os
  state=1
  for com in sys.argv:
    if com in ['help','clean']: state=0
  bxtarget='../build/lib'
  bptarget='../build/lib/CGNS'  
  if (not os.path.exists(bxtarget)):
    os.makedirs(bxtarget)
    pt=distutils.util.get_platform()
    vv="%d.%d"%(sys.version_info[0],sys.version_info[1])
    tg="%s/../build/lib.%s-%s/CGNS"%(os.getcwd(),pt,vv)
    lg="%s/../build/lib/CGNS"%(os.getcwd())
    os.makedirs(tg)
    os.symlink(tg,lg)
  oldsyspath=sys.path
  sys.path=['../lib']
  cfgdict={}
  import pyCGNSconfig_default as C_D
  sys.path=oldsyspath
  for ck in dir(C_D):
    if (ck[0]!='_'): cfgdict[ck]=C_D.__dict__[ck]
  pg=prodtag()
  cfgdict['DATE']=pg[0]
  cfgdict['PLATFORM']="%s %s %s"%(pg[1][0],pg[1][1],pg[1][-1])
  updateConfig('..',bptarget,cfgdict)
  sys.path=[bptarget]+sys.path
  try:
    import pyCGNSconfig as C
    if ('HDF5' in deps):
      (C.HDF5_VERSION,
       C.HDF5_PATH_INCLUDES,
       C.HDF5_PATH_LIBRARIES,
       C.HDF5_LINK_LIBRARIES,
       C.HDF5_EXTRA_ARGS)=find_HDF5(C.HDF5_PATH_INCLUDES+C.INCLUDE_DIRS,
                                    C.HDF5_PATH_LIBRARIES+C.LIBRARY_DIRS,
                                    C.HDF5_LINK_LIBRARIES)
    if ('MLL' in deps):
      (C.MLL_VERSION,
       C.MLL_PATH_INCLUDES,
       C.MLL_PATH_LIBRARIES,
       C.MLL_LINK_LIBRARIES,
       C.MLL_EXTRA_ARGS)=find_MLL(C.MLL_PATH_INCLUDES+C.INCLUDE_DIRS,
                                  C.MLL_PATH_LIBRARIES+C.LIBRARY_DIRS,
                                  C.MLL_LINK_LIBRARIES,
                                  C.MLL_EXTRA_ARGS)
    if ('CHLone' in deps):
      (C.CHLONE_VERSION,
       C.CHLONE_PATH_INCLUDES,
       C.CHLONE_PATH_LIBRARIES,
       C.CHLONE_LINK_LIBRARIES,
       C.CHLONE_EXTRA_ARGS)=find_CHLone(C.CHLONE_PATH_INCLUDES+C.INCLUDE_DIRS,
                                        C.CHLONE_PATH_LIBRARIES+C.LIBRARY_DIRS,
                                        C.CHLONE_LINK_LIBRARIES)
    if ('numpy' in deps):
      (C.NUMPY_VERSION,
       C.NUMPY_PATH_INCLUDES,
       C.NUMPY_PATH_LIBRARIES,
       C.NUMPY_LINK_LIBRARIES,
       C.NUMPY_EXTRA_ARGS)=find_numpy(C.NUMPY_PATH_INCLUDES+C.INCLUDE_DIRS,
                                      C.NUMPY_PATH_LIBRARIES+C.LIBRARY_DIRS)
  except ImportError:
    print 'pyGCNS[ERROR]: %s setup cannot find pyCGNSconfig.py file!'%tag
    sys.exit(1)
  return (C, state)

# --------------------------------------------------------------------
def installConfigFiles():
  lptarget='..'
  bptarget='./build/lib/CGNS'  
  for ff in rootfiles:
    shutil.copy("%s/lib/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))
  for ff in compfiles:
    shutil.copy("%s/lib/compatibility/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything
relist=['^.*~$','^core\.*$','^pyCGNS\.log\..*$',
        '^#.*#$','^.*\.aux$','^.*\.pyc$','^.*\.bak$','^.*\.l2h',
        '^Output.*$']
reclean=[]

for restring in relist:
  reclean.append(re.compile(restring))

def wselect(args,dirname,names):
  for n in names:
    for rev in reclean:
      if (rev.match(n)):
        # print "%s/%s"%(dirname,n)
        os.remove("%s/%s"%(dirname,n))
        break

class clean(_clean):
  def walkAndClean(self):
    os.path.walk("..",wselect,[])
  def run(self):
    import glob
    rdirs=glob.glob("./build/*")
    for d in rdirs: remove_tree(d)
    if os.path.exists("./build"):     os.remove("./build")
    if os.path.exists("./Doc/_HTML"): remove_tree("./Doc/_HTML")
    if os.path.exists("./Doc/_PS"):   remove_tree("./Doc/_PS")
    if os.path.exists("./Doc/_PDF"):  remove_tree("./Doc/_PDF")
    self.walkAndClean()

# --------------------------------------------------------------------
def confValueAsStr(v):
  if (type(v)==type((1,))): return str(v)
  if (type(v)==type([])):   return str(v)
  if (v in [True,False]):   return str(v)
  else:                     return '"%s"'%str(v)

# --------------------------------------------------------------------
def updateConfig(pfile,gfile,config_default):
  if (not os.path.exists("%s/pyCGNSconfig.py"%(gfile))):
    print "### pyCGNS: create new pyCGNSconfig.py file"
    newconf=1
  else:
    f1=os.stat("%s/pyCGNSconfig.py"%(gfile))
    f2=os.stat("%s/pyCGNSconfig_user.py"%(pfile))
    if (f1.st_mtime < f2.st_mtime):
      newconf=1
      print "### pyCGNS: use modified pyCGNSconfig_user.py file"
    else:
      newconf=0
      print "### pyCGNS: use existing pyCGNSconfig.py file"
  if newconf:
    sys.path=['..']+sys.path
    import pyCGNSconfig_user
    for ck in dir(pyCGNSconfig_user):
      if (ck[0]!='_'): config_default[ck]=pyCGNSconfig_user.__dict__[ck]
    f=open("%s/pyCGNSconfig.py"%(gfile),'w+')
    f.writelines(config_default['file_pattern']%config_default)
    f.close()

# --------------------------------------------------------------------
def find_HDF5(pincs,plibs,libs):
  notfound=1
  extraargs=[]
  vers=''
  for pth in plibs:
    if (    (os.path.exists(pth+'/libhdf5.a'))
         or (os.path.exists(pth+'/libhdf5.so'))
         or (os.path.exists(pth+'/libhdf5.sl'))):
      notfound=0
      break
  if notfound:
    print "### pyCGNS: ERROR: libhdf5 not found, please check paths:"
    print "### pyCGNS: ",plibs
    return ([],)*5
  notfound=1
  for pth in pincs:
    if (os.path.exists(pth+'/hdf5.h')): notfound=0
  if notfound:
    print "### pyCGNS: ERROR: hdf5.h not found, please check paths"
    print "### pyCGNS: ",pincs
    return ([],)*5
  ifh='HDF5 library version: unknown'
  for pth in pincs:
    if (os.path.exists(pth+'/H5public.h')):
      fh=open(pth+'/H5public.h','r')
      fl=fh.readlines()
      fh.close()
      found=0
      for ifh in fl:
        if (ifh[:21] == "#define H5_VERS_INFO "):
          vers=ifh.split('"')[1]
          found=1
      if found: break
      print "### pyCGNS: ERROR: cannot find hdf5 version, please check paths"
      print "### pyCGNS: ",pincs
      return ([],)*5
  return (vers,pincs,plibs,libs,extraargs)

# --------------------------------------------------------------------
def find_MLL(pincs,plibs,libs,extraargs):
  notfound=1
  vers=''
  libs=['cgns','hdf5']
  for pth in pincs:
    if (os.path.exists(pth+'/cgnslib.h')):
      notfound=0
      f=open(pth+'/cgnslib.h','r')
      l=f.readlines()
      f.close()
      for ll in l:
        if (ll[:20]=="#define CGNS_VERSION"):
          cgnsversion=ll.split()[2]
          if (cgnsversion<'3000'):
            print "### pyCGNS: ERROR: version should be v3.x for MLL"
            return ([],)*5
      break
  if notfound:
    print "### pyCGNS: ERROR: cgnslib.h not found, please check paths"
    print "### pyCGNS: ",pincs
    return ([],)*5

  notfound=1
  for pth in plibs:
    if (    (os.path.exists(pth+'/libcgns.a'))
         or (os.path.exists(pth+'/libcgns.so'))
         or (os.path.exists(pth+'/libcgns.sl'))):
      cgnslib='cgns'
      notfound=0
      break
  if notfound:
    print "### pyCGNS: ERROR: libcgns not found, please check paths:"
    print "### pyCGNS: ",plibs
    return ([],)*5
    
  notfound=1
  for pth in pincs:
    if (os.path.exists(pth+'/adfh/ADFH.h')):
      extraargs+=['-D__ADF_IN_SOURCES__']
      notfound=0
    if (os.path.exists(pth+'/ADFH.h')):
      notfound=0

  if notfound:
    print "### pyCGNS: Warning: ADFH.h not found, using pyCGNS own headers"
    extraargs+=['-U__ADF_IN_SOURCES__']
  #extraargs+=['-DLEGACY_SUPPORT']

  return (cgnsversion,pincs,plibs,libs,extraargs)

# --------------------------------------------------------------------
def find_CHLone(pincs,plibs,libs):
  extraargs=[]
  vers=''
  notfound=1
  libs=['CHLone']
  for pth in plibs:
    if (    (os.path.exists(pth+'/libCHlone.a'))
         or (os.path.exists(pth+'/libCHlone.so'))
         or (os.path.exists(pth+'/libCHlone.sl'))):
      libs=['CHlone']
      notfound=0
      break
  if notfound:
    print "### pyCGNS: ERROR: libCHlone not found, please check paths:"
    print "### pyCGNS: ",plibs
    return ([],)*5
  notfound=1      
  for pth in pincs:
    if (os.path.exists(pth+'/CHLone/CHlone.h')): notfound=0
  if notfound:
    print "### pyCGNS: ERROR: CHLone/CHlone.h not found, please check paths"
    print "### pyCGNS: ",pincs
    return ([],)*5
  
  return (vers,pincs,plibs,libs,extraargs)

# --------------------------------------------------------------------
def find_numpy(pincs,plibs):
  vers='1.4'
  extraargs=[]
  libs=[]
  pdir=os.path.normpath(sys.prefix)
  xdir=os.path.normpath(sys.exec_prefix)
  pincs=['%s/lib/python%s/site-packages/numpy/core/include'\
         %(xdir,sys.version[:3])]
  return (vers,pincs,plibs,libs,extraargs)

# --- last line

