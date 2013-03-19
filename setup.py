#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import os
import sys
import string
import getopt
sys.path=['./lib']+sys.path
import setuputils

# hg pull ssh://poinot@hg.code.sf.net/p/pycgns/code
#
# OLD (bad) ssh://poinot@pycgns.hg.sourceforge.net/hgroot/pycgns/pycgns
pcom=sys.executable

# order IS significant
CGNSmodList=['MAPper','WRApper','PATternMaker','NAVigater',
             'DATaTracer','VALidater','APPlicationSampler']

modList=CGNSmodList[:]
modList.remove('DATaTracer')

solist='m:'
lolist=["without-mod=","single-mod=","prefix=","force"]

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:],solist,lolist)
except getopt.GetoptError:
  print 72*'='
  print "### pyCGNS: ERROR"
  print "###       : see documentation for details on installation options"
  print "###       : in particular for default values."
  print "###       : known options are: "
  print "###       :   --without-mod='MMM' to install without this module"
  print "###       :   --single-mod='MMM'  to install only this module"
  print "###       :   --prefix='/path'    to install in the specified path"
  print "###       :   --force             force all rebuilds, update version"
  sys.exit(2)

for o,a in opts:
  m=a
  for mm in CGNSmodList:
    if (mm[:3] == a):
      m=mm
      break
  if ((o == "--without-mod") and (m in CGNSmodList)): modList.remove(m)
  if ((o == "--single-mod")  and (m in CGNSmodList)): modList=[m]
  if (o == "--force"):
    os.system('hg parents --template="{rev}\n" > ./lib/revision.tmp')
    setuputils.updateVersionInFile('./lib/pyCGNSconfig_default.py')
  
modArgs=[]
for opt in sys.argv[1:]:
  if (opt[:12] not in ['--without-mod','--single-mod']): modArgs.append(opt)
modArgs=string.join(modArgs)

for mod in modList:
  print '\n',mod, (69-len(mod))*'-'
  if os.path.exists('./%s/setup.py'%mod):
    os.chdir(mod)
    com='%s setup.py %s'%(pcom,modArgs)
    print com
    os.system(com)
    os.chdir('..')

print '\n', 69*'-'
  
# --- last line

