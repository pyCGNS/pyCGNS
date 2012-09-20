import CGNS.WRA.mll as Mll
import numpy as N
import CGNS.PAT.cgnskeywords as CK

# ----------------------------------------------------------------------
def acube(im=3,jm=5,km=7,offset=0):
  # inverse k/i in order to get correct order in ADF file
  x=N.zeros((km,jm,im),'d')
  y=N.zeros((km,jm,im),'d')
  z=N.zeros((km,jm,im),'d')
  for i in range(im):
    for j in range(jm):
      for k in range(km):
        x[k,j,i]=i+(im-1)*offset
        y[k,j,i]=j
        z[k,j,i]=k
  return (x,y,z)

c01=acube()
c02=acube(offset=1)
c03=acube(offset=2)

# ------------------------------------------------------------------------

a=Mll.pyCGNS('tmp/testmll36.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.zone_write(1,'Zone 01',N.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.zone_write(1,'Zone 02',N.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.zone_write(1,'Zone 03',N.array([[3,5,7],[2,4,6],[0,0,0]]),CK.Structured)
a.coord_write(1,1,CK.RealDouble,CK.CoordinateX_s,c01[0])
a.coord_write(1,1,CK.RealDouble,CK.CoordinateY_s,c01[1])
a.coord_write(1,1,CK.RealDouble,CK.CoordinateZ_s,c01[2])
a.coord_write(1,2,CK.RealDouble,CK.CoordinateX_s,c02[0])
a.coord_write(1,2,CK.RealDouble,CK.CoordinateY_s,c02[1])
a.coord_write(1,2,CK.RealDouble,CK.CoordinateZ_s,c02[2])
a.coord_write(1,3,CK.RealDouble,CK.CoordinateX_s,c03[0])
a.coord_write(1,3,CK.RealDouble,CK.CoordinateY_s,c03[1])
a.coord_write(1,3,CK.RealDouble,CK.CoordinateZ_s,c03[2])
a.sol_write(1,1,"Initialize",CK.CellCenter)
a.sol_write(1,1,"Result",CK.CellCenter)
a.sol_write(1,2,"Initialize",CK.CellCenter)
a.sol_write(1,2,"Result",CK.CellCenter)
a.sol_write(1,3,"Initialize",CK.CellCenter)
a.sol_write(1,3,"Result",CK.CellCenter)
a.boco_write(1,1,'BC',12,4,2,N.array([[1,1,1],[3,5,1]]))
a.field_write(1,1,1,3,'data array1',N.ones((2,4,6))*1.2)
a.field_write(1,1,1,4,'data array2',N.ones((2,4,6))*1.2)
a.field_write(1,1,1,2,'data array3',N.ones((2,4,6)))

p=a.rigid_motion_write(1,1,'rigid1',3)
#p=a.rigid_motion_write(1,1,'rigid2',2)
print p

a.close()
