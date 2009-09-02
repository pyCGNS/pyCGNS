# -------------------------------------------------------------------------
# pyCGNS - CFD General Notation System - SIDS-to-Python MAPping            
# $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $         
# See license file in the root directory of this Python module source      
# -------------------------------------------------------------------------
from  distutils.core import setup, Extension
from  distutils.util import get_platform



# --- pyCGNSconfig search
import sys
sys.path+=['..']
import setuputils
(pyCGNSconfig,installprocess)=setuputils.search('MAP')
# ---
