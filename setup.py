#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $File$
#  $Node$
#  $Last$
#  -------------------------------------------------------------------------
import os
import sys
import string
import getopt

pcom=sys.executable

version=4
versionList=[4]

# order IS significant
CGNSmodList=['MAPper','WRApper','PATternMaker','NAVigater',
             'DATaTracer','VALidater','APPlicater']
modList=CGNSmodList[:]

solist='m:'
lolist=["without-mod=","single-mod=","prefix="]

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:],solist,lolist)
except getopt.GetoptError:
  print 72*'='
  print "### pyCGNS: ERROR"
  print "### pyCGNS: see documentation for details on installation options"
  print "### pyCGNS: in particular for default values."
  print "### pyCGNS: known options are: "
  print "### pyCGNS:   --without-mod='MMM' to install without this module"
  print "### pyCGNS:   --single-mod='MMM'  to install only this module"
  print "### pyCGNS:   --prefix='/path'    to install in the specified path"
  sys.exit(2)

for o,a in opts:
  m=a
  for mm in CGNSmodList:
    if (mm[:3] == a):
      m=mm
      break
  if ((o == "--without-mod") and (m in CGNSmodList)): modList.remove(m)
  if ((o == "--single-mod")  and (m in CGNSmodList)): modList=[m]
  
modArgs=[]
for opt in sys.argv[1:]:
  if (opt[:12] not in ['--without-mod','--single-mod']): modArgs.append(opt)
modArgs=string.join(modArgs)

for mod in modList:
  print '\n',mod, 65*'-'
  if os.path.exists('./%s/setup.py'%mod):
    os.chdir(mod)
    com='%s setup.py %s'%(pcom,modArgs)
    print com
    os.system(com)
    os.chdir('..')

print '\n', 69*'-'
  
# --- last line

