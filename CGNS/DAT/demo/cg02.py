#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

import CGNS
import CGNS.DAT.demo.cgutils as cgu

#
dbid1 = cgu.getLocalId("D01")
db = CGNS.pyCGNS('%s.cgns' % dbid1, CGNS.MODE_WRITE)
bi = db.basewrite(dbid1, 3, 3)
db.close()
#
dbid2 = cgu.getLocalId("D01")
db = CGNS.pyCGNS('%s.cgns' % dbid2, CGNS.MODE_WRITE)
bi = db.basewrite(dbid2, 3, 3)
db.close()
#
db = CGNS.pyCGNS('%s.cgns' % dbid1, CGNS.MODE_MODIFY)
for z in ["domain%.5d" % d for d in range(1, 5)]:
    zi = db.zonewrite(bi, z, (11, 23, 5, 10, 22, 4, 0, 0, 0), CGNS.Structured)
    cgu.addGrids(db, bi, zi)
db.close()
#
db = CGNS.pyCGNS('%s.cgns' % dbid2, CGNS.MODE_MODIFY)
cgu.mapZones(db, '%s.cgns' % dbid1)
cgu.linkGrids(db, '%s.cgns' % dbid1)
for zi in range(1, db.nzones(1) + 1):
    cgu.addConservatives(db, bi, zi, ['FlowSolution#EndOfRun'])
db.close()
#
