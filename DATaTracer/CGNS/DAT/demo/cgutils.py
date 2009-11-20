#
#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

import CGNS
import NumArray
#
# db : a valid and open pyCGNS instance
#
def addConservatives(db,bid,zid,solnames):
  for s in solnames:  
    sid=db.solwrite(bid,zid,s,CGNS.CellCenter)
    zinfo=db.zoneread(bid,zid)
    for f in [CGNS.MomentumX,
              CGNS.MomentumY,
              CGNS.MomentumZ,
              CGNS.Density,
              CGNS.EnergyStagnationDensity]:
      data=NumArray.ones((zinfo[3][0],zinfo[3][1],zinfo[3][2]),NumArray.Float)
      db.fieldwrite(bid,zid,sid,CGNS.RealDouble,f,data)
#
#
def addGrids(db,bid,zid):
  zinfo=db.zoneread(bid,zid)
  data=NumArray.ones((zinfo[3][0],zinfo[3][1],zinfo[3][2]),NumArray.Float)
  for cn in [CGNS.CoordinateX,
             CGNS.CoordinateY,
             CGNS.CoordinateZ]:
    db.coordwrite(bid,zid,CGNS.RealDouble,cn,data)
#
#
def mapZones(db,file):
  dbx=CGNS.pyCGNS(file,CGNS.MODE_READ)
  for zid in range(1,dbx.nzones(1)+1): # bet we have a single base... 
    zinfo=dbx.zoneread(1,zid)
    db.zonewrite(1,zinfo[2],zinfo[3],CGNS.Structured)
  dbx.close()
#    
#
def linkGrids(db,file):
  dbx=CGNS.pyCGNS(file,CGNS.MODE_READ)
  for zid in range(1,dbx.nzones(1)+1): # bet we have a single base... 
    zinfo=dbx.zoneread(1,zid)
    gn="/%s/%s/GridCoordinates"%(file[:-5],zinfo[2])
    db.goto(1,[(CGNS.Zone_t,zid)])
    db.linkwrite("GridCoordinates",file,gn)
  dbx.close()  
#
def getLocalId(pfx=""):
  import time
  nid=time.time()*10e6 # -long(time.time()/10e3)*10e3
  sid=hex(long(nid))[4:][2:-1]
  if pfx:
    r="%s:%s"%(pfx,sid)
  else:
    r=sid
  return r
#
