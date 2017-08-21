#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from CGNS.APP.lib import mergeTrees
import CGNS.MAP as CGM


def treeLinkZGC(T1):
    pass


def fileLinkZGC(*files):
    l = []
    for f in files:
        l += [CGM.load(f)[0]]
