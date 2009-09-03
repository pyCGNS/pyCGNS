# =============================================================================
# pyCGNS - CFD General Notation System - ONERA - marc.poinot@onera.fr
# $Rev: 79 $ $Date: 2009-03-13 10:19:54 +0100 (Fri, 13 Mar 2009) $
# See file 'license' in the root directory of this Python module source 
# tree for license information. 
# =============================================================================

# compatibility methods
try:
  import CGNS.WRA as WRA
  import CGNS.WRA.utils as utils
  import CGNS.WRA.midlevel as midlevel
  import CGNS.WRA.wrapper as wrap
except ImportError:
  pass
#
