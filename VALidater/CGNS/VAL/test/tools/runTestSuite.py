#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import glob
import sys
import CCCCC.tools.ccccc
import CCCCC.utils.stuff as ustf
import posixpath

# CCCCC should be installed
# should be run here, in the local directory
# this tool is not for end-users

# test files are *.xml
# a file starting with gXX is to be run with grammar XX

l_ok    =glob.glob('../files.ok/*.xml')
l_failed=glob.glob('../files.failed/*.xml')

def getGrammar(f):
  fx=posixpath.basename(f)
  print fx
  if (fx[0] == 'g'):
    g="g%s%s.rng"%(fx[1],fx[2])
    sys.argv[1:]=['-r']  
    sys.argv[1:]=[g]
  
print "### C5 Test suite - Log file in /tmp/c5test.log"

fout=open("/tmp/c5test.log",'w+')
sys.stdout=fout

for l in l_ok:
  print 70*'-'
  sys.argv[1:]=[l]
  getGrammar(l)
  print "ARGS", sys.argv
  try:
    s="### %s should succeed ["%posixpath.basename(l)
    CCCCC.tools.ccccc.go(1)
    s+="yes]\n"
    sys.stderr.write(s)
    print s
  except ustf.c5Exception:
    s+="NO]\n"
    sys.stderr.write(s)
    print s

for l in l_failed:
  print 70*'-'
  sys.argv[1:]=[l]
  sys.argv[1:]=[l]
  getGrammar(l)
  print "ARGS", sys.argv
  try:
    s="### %s should failed  ["%posixpath.basename(l)
    CCCCC.tools.ccccc.go(1)
    s+="NO]\n"
    sys.stderr.write(s)
    print s
  except ustf.c5Exception:
    s+="yes]\n"
    sys.stderr.write(s)
    print s

print 70*'-'
fout.close()
