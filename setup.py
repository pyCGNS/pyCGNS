# =============================================================================
# pyCGNS - CFD General Notation System - ONERA - marc.poinot@onera.fr
# $Rev: 79 $ $Date: 2009-03-13 10:19:54 +0100 (Fri, 13 Mar 2009) $
# See file 'license' in the root directory of this Python module source 
# tree for license information. 
# =============================================================================

import os
import sys
import string
import getopt

version=4
versionList=[4]

# order IS significant
CGNSmodList=['MAPper','WRApper','PATternMaker','NAVigater','DATaTracer',
             'VALidater','TRAnslater']
modList=CGNSmodList[:]

solist='m:'
lolist=["without-mod=","single-mod="]

try:
  opts, args = getopt.gnu_getopt(sys.argv[1:],solist,lolist)
except getopt.GetoptError:
  print 72*'='
  print "### pyCGNS: see README file for details on installation options"
  print "### pyCGNS: in particular for default values."
  sys.exit(2)

for o,a in opts:
  if ((o == "--without-mod") and (a in CGNSmodList)): modList.remove(a)
  if ((o == "--single-mod")  and (a in CGNSmodList)): modList=[a]
  
modArgs=[]
for opt in sys.argv[1:]:
  if (opt[:12] not in ['--without-mod','--single-mod']): modArgs.append(opt)
modArgs=string.join(modArgs)

for mod in modList:
  print '\n',mod, 65*'-'
  if os.path.exists('./%s/setup.py'%mod):
    os.chdir(mod)
    com='python setup.py %s'%modArgs
    print com
    os.system(com)
    os.chdir('..')

print '\n', 69*'-'
  
# --- last line

