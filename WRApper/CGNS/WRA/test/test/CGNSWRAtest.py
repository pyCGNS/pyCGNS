# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 61 $ $Date: 2008-09-09 08:58:45 +0200 (Tue, 09 Sep 2008) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#

# test all the stuff we have there

from CGNS import pyCGNSconfig
import string

def showConfig():
    from CGNS.version import __vid__
    print '-'*70
    print " pyCGNS v%s"%__vid__
    print '-'*70
    print " build on %s\n for %s "%(pyCGNSconfig.DATE,
                                    string.join([pyCGNSconfig.PLATFORM[0],
                                                 pyCGNSconfig.PLATFORM[2],
                                                 pyCGNSconfig.PLATFORM[4],]))
    print '-'*70
    if (pyCGNSconfig.HAS_MLL):
        print " MLL  version  : %s"%pyCGNSconfig.MLL_VERSION
        print " MLL  path     : %s"%pyCGNSconfig.PATH_MLL
    else:
        print " MLL           : NO"
        
    if (pyCGNSconfig.HAS_CHLONE):
        print " CHlone version: %s"%pyCGNSconfig.HAS_CHLONE
        print " CHlone path   : %s"%pyCGNSconfig.PATH_CHLONE
    else:
        print " CHlone        : NO"

    print '-'*70

def run(path):
  import sys
  sys.path=path
  if 1: #try:
    print '-'*70
    print "### *** ADFextensive"
    import ADFextensive
#  except:
    print "### *** FAILED ADFextensive"


  try:
    print '-'*70
    print "### *** MLLextensive"
    import CGNS.test.MLLextensive
  except:
    print "### *** FAILED MLLextensive"

  try:
    print '-'*70
    print "### *** HDFextensive"
    import CGNS.test.HDFextensive
  except:
    print "### *** FAILED hdfextensive"

print '-'*70

