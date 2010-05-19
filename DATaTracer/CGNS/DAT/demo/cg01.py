#!/usr/bin/env python
# $Id: cg01.py 19 2003-06-18 12:29:31Z mpoinot $
#
# Basic required tree -
#
import CGNS
import DAX.demo.cgutils as cgu
#
dbid1=cgu.getLocalId("D01")
db=CGNS.pyCGNS('%s.cgns'%dbid1,CGNS.MODE_WRITE)
bi=db.basewrite(dbid1,3,3)
db.close()
#
db=CGNS.pyCGNS('%s.cgns'%dbid1,CGNS.MODE_MODIFY)
for z in [ "domain%.5d"%d for d in range(1,5)]:
  zi=db.zonewrite(bi,z,(11,23,5,10,22,4,0,0,0),CGNS.Structured)
  cgu.addGrids(db,bi,zi)
db.close()
#
