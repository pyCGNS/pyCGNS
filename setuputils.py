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

rootfiles=['errors.py','version.py']
compfiles=['__init__.py','midlevel.py','wrap.py']

# --------------------------------------------------------------------
def search(tag):
  import sys
  import distutils.util
  import os
  state=1
  for com in sys.argv:
    if com in ['help','clean']: state=0
  bxtarget='../build/lib'
  bptarget='../build/lib/CGNS'  
  pfile='../pyCGNSconfig.py.in'
  if (not os.path.exists(bxtarget)):
    os.makedirs(bxtarget)
    pt=distutils.util.get_platform()
    vv="%d.%d"%(sys.version_info[0],sys.version_info[1])
    tg="%s/../build/lib.%s-%s/CGNS"%(os.getcwd(),pt,vv)
    lg="%s/../build/lib/CGNS"%(os.getcwd())
    os.makedirs(tg)
    os.symlink(tg,lg)
  cfgdict={}
  if (state): cfgdict=findProductionContext()
  updateConfig('..',bptarget,cfgdict)
  sys.path+=[bptarget]
  try:
    import pyCGNSconfig
  except ImportError:
    print 'pyGCNS[ERROR]: %s setup cannot find pyCGNSconfig.py file!'%tag
    sys.exit(1)
  return (pyCGNSconfig, state)

# --------------------------------------------------------------------
def findProductionContext():
  return {}

# --------------------------------------------------------------------
def installConfigFiles():
  lptarget='..'
  bptarget='./build/lib/CGNS'  
  for ff in rootfiles:
    shutil.copy("%s/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))
    print "%s/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff)
  for ff in compfiles:
    shutil.copy("%s/compatibility/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff))
    print "%s/compatibility/%s"%(lptarget,ff),"%s/%s"%(bptarget,ff)    

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
    if os.path.exists("./build"):     remove_tree("./build")
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
def updateConfig(pfile,gfile,config):
  if (not os.path.exists("%s/pyCGNSconfig.py"%(gfile))):
    f=open("%s/pyCGNSconfig.py.in"%(pfile),'r')
    cf=f.readlines()
    f.close()
  else:
    f=open("%s/pyCGNSconfig.py"%(gfile),'r')
    cf=f.readlines()
    f.close()
  rl=[]
  ck=config.keys()
  for l in cf:
    found=0
    for c in ck:
      if (((len(c))<=len(l)) and (c==l[:len(c)]) and (l[-2]=='#')):
        rl+=['%s=%s#\n'%(c,confValueAsStr(config[c]))]
        found=1
    if not found:
        rl+=l
  f=open("%s/pyCGNSconfig.py"%(gfile),'w+')
  f.writelines(rl)
  f.close()
  
# --- last line

