# CFD General Notation System - CGNS lib wrapper
# ONERA/DSNA/ELSA - poinot@onera.fr
# pyCGNS - $Rev: 56 $ $Date: 2008-06-10 09:44:23 +0200 (Tue, 10 Jun 2008) $
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
# -----------------------------------------------------------------------------
# See file COPYING in the root directory of this Python module source 
# tree for license information. 

import CGNS.cgnserrors   as E
from CGNS import pyCGNSconfig

try:
  if (not pyCGNSconfig.HAS_CHLONE):
    import sys
    print "### pyCGNS: NOT built with HDF/Python binding"
    print "### pyCGNS: Cannot perform HDF tests"
    sys.exit(1)
except KeyError:
    print "### pyCGNS: Cannot find pyCGNS config ?"
    sys.exit(1)

import CGNS.hdf
import numpy as N
import CGNS.cgnshdf as H
import CGNS.cgnslib as L
import CGNS.cgnskeywords as K

# ----------------------------------------------------------------------
def tt(s):
  print "#---#"
  print s

print "-" *70
print "#   # HDF Trivial calls to every API function"

tt("# 01# Base")
b1=L.newBase(None,"Base 001",3,3)
b2=L.newBase(None,"Base 002",3,3)
b3=L.newBase(None,"Base 003",2,2)
b4=L.newBase(None,"Base 004",3,3)

tt("# 02# Zone")
L.newZone(b3,"Zone 001")
L.newZone(b3,"Zone 007")

tt("# 03# Add node")
db=H.CGNSHDF5tree("./T01.hdf",CGNS.hdf.NEW)
db.flags|=CGNS.hdf.TRACE
db.addNode(b1)
db.addNode(b2)
db.addNode(b3)
db.save(b3)
db.addNode(b4)
db.close()

tt("# 04# Close")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.READ)
db.flags|=CGNS.hdf.TRACE
db.close()

from CGNS.pattern import Zone_t
from CGNS.pattern import GridCoordinates_t

cx=N.ones((5,3,9),dtype=N.float64)
cy=N.ones((5,3,9),dtype=N.float64)
cz=N.ones((5,3,9),dtype=N.float64)

tt("# 05# Update")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.READ)
tz=Zone_t.data
L.updateZone(tz,size=(5,3,9))
g1=L.hasChildName(tz,K.GridCoordinates_s)
L.newRind(g1,N.array([0,0,0,0,1,1]))
L.newDataClass(g1)
L.newDimensionalUnits(g1)
L.newUserDefinedData(g1,'{UserDefinedData}')
L.newDescriptor(g1,'{Descriptor}')
L.newDataArray(g1,K.CoordinateX_s,cx)
L.newDataArray(g1,K.CoordinateY_s,cy)
L.newDataArray(g1,K.CoordinateZ_s,cz)

tt("# 05# Add tree")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.NEW)
db.flags|=CGNS.hdf.TRACE
bid=db.addNode(b1)
zid=db.addTree(tz,bid)
db.close()

tt("# 06# Load")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.READ)
print 70*'='
print db.load()
db.close()

db=H.CGNSHDF5tree("./SquaredNozzle-05.hdf",CGNS.hdf.READ)
t=db.load()
db.close()

tt("# 07# Save")
db=H.CGNSHDF5tree("./SQNZ.hdf",CGNS.hdf.NEW)
db.save(t)
db.close()

tt("# 08# Find")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.READ)
id=db.find(None,"/Base 001/{Zone}/ZoneBC/{BC}")
db.load("/Base 001/{Zone}/ZoneBC/{BC}")
db.close()

tt("# 09# Find")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.UPDATE)
id=db.find(None,"/Base 001/{Zone}/ZoneBC")
db.move(id,'./{BC}',id,'./BC01')#01
db.move(id,'./{BC}',id,'./BC01')#02
db.move(id,'./{BC}',id,'./BC01')#03
db.move(id,'./{BC}',id,'./BC01')#04
print 70*'='
print db.load('/Base 001/{Zone}/ZoneBC/')
db.close()

tt("# 10# Delete")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.UPDATE)
id=db.find(None,"/Base 001/{Zone}/ZoneBC")
db.delete(id,'BC01')
db.close()

tt("# 11# Link")
db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.UPDATE)
id=db.find(None,"/Base 001")
print 70*'='
print db.load("/Base 001/{Zone}/GridCoordinates/DimensionalUnits")
print 70*'='
db.link(id,"DimensionalUnits",
        '',"/Base 001/{Zone}/GridCoordinates/DimensionalUnits")
print 70*'='
id=db.find(None,"/Base 001/{Zone}/GridCoordinates/DimensionalUnits")
print 70*'='
print db.retrieve(id)
print 70*'='
db.close()

db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.UPDATE)
db.link(None,"Base 999","T01.hdf","/Base 003")
db.close()

db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.READ)
print db.load()
db.close()

db=H.CGNSHDF5tree("./T02.hdf",CGNS.hdf.UPDATE)
id=db.find(None,"/Base 001/{Zone}/GridCoordinates/DimensionalUnits")
db.update([id,None,None,None,None])
print db.retrieve(id)
db.close()
