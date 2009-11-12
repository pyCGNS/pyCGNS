# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 35 $ $Date: 2007-12-14 14:54:02 +0100 (Fri, 14 Dec 2007) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
import CGNS.PAT.cgnslib      as C
import CGNS.PAT.cgnserrors   as E
import CGNS.PAT.cgnskeywords as K
import numpy             as N

data=['ZoneIterativeData',None,[],'ZoneIterativeData_t']
status=''
comment=''
pattern=[data, status, comment]
