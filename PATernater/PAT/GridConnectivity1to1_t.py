# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 35 $ $Date: 2007-12-14 14:54:02 +0100 (Fri, 14 Dec 2007) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.cgnslib      as C
import CGNS.cgnserrors   as E
import CGNS.cgnskeywords as K
import numpy             as N

data=['GridConnectivity1to1',None,[],'GridConnectivity1to1_t']
status=''
comment=''
pattern=[data, status, comment]
