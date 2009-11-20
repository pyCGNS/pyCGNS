#  -------------------------------------------------------------------------
#  pyCGNS.APP - Python package for CFD General Notation System - APPlicater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
from  distutils.core import setup, Extension
from  distutils.util import get_platform



# --- pyCGNSconfig search
import sys
sys.path+=['../lib']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('TRA')
# ---
