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
#from  distutils.command.install import install as _install

rootfiles=['pyCGNSconfig.py','errors.py','version.py']
compfiles=['__init__.py','midlevel.py','wrap.py']
def installConfigFiles(searchpath):
  bptarget='./build/lib/CGNS'
  bxtarget='./build/lib.%s-%s/CGNS'%(get_platform(),sys.version[0:3])
  for d in searchpath:
    if (os.path.exists("%s/pyCGNSconfig.py"%d)):
      try:
        if (not os.path.exists(bptarget)): os.makedirs(bptarget)
        if (not os.path.exists(bxtarget)): os.makedirs(bxtarget)
      except os.error: pass
      for ff in rootfiles:
        shutil.copy("%s/%s"%(d,ff),"%s/%s"%(bptarget,ff))
        shutil.copy("%s/%s"%(d,ff),"%s/%s"%(bxtarget,ff))
      for ff in compfiles:
        shutil.copy("%s/compatibility/%s"%(d,ff),"%s/%s"%(bptarget,ff))
        shutil.copy("%s/compatibility/%s"%(d,ff),"%s/%s"%(bxtarget,ff))

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
    os.path.walk(".",wselect,[])
  def run(self):
    if os.path.exists("./build"):     remove_tree("./build")
    if os.path.exists("./Doc/_HTML"): remove_tree("./Doc/_HTML")
    if os.path.exists("./Doc/_PS"):   remove_tree("./Doc/_PS")
    if os.path.exists("./Doc/_PDF"):  remove_tree("./Doc/_PDF")
    self.walkAndClean()

