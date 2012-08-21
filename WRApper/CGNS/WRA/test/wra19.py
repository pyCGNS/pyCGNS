import CGNS.WRA.mll as Mll
import numpy as N


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

a=Mll.pyCGNS('tmp/testmll19.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,3)
a.zone_write(1,'Zone 01',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.zone_write(1,'Zone 02',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.zone_write(1,'Zone 03',N.array([[3,5,7],[2,4,6],[0,0,0]]),Mll.Structured)
a.coord_write(1,1,Mll.RealDouble,Mll.CoordinateX,c01[0])
a.coord_write(1,1,Mll.RealDouble,Mll.CoordinateY,c01[1])
a.coord_write(1,1,Mll.RealDouble,Mll.CoordinateZ,c01[2])
a.coord_write(1,2,Mll.RealDouble,Mll.CoordinateX,c02[0])
a.coord_write(1,2,Mll.RealDouble,Mll.CoordinateY,c02[1])
a.coord_write(1,2,Mll.RealDouble,Mll.CoordinateZ,c02[2])
a.coord_write(1,3,Mll.RealDouble,Mll.CoordinateX,c03[0])
a.coord_write(1,3,Mll.RealDouble,Mll.CoordinateY,c03[1])
a.coord_write(1,3,Mll.RealDouble,Mll.CoordinateZ,c03[2])
a.sol_write(1,1,"Initialize",Mll.CellCenter)
a.sol_write(1,1,"Result",Mll.CellCenter)
a.sol_write(1,2,"Initialize",Mll.CellCenter)
a.sol_write(1,2,"Result",Mll.CellCenter)
a.sol_write(1,3,"Initialize",Mll.CellCenter)
a.sol_write(1,3,"Result",Mll.CellCenter)
a.boco_write(1,1,'BC',12,4,2,N.array([[1,1,1],[3,5,1]]))
a.boco_write(1,1,'BC2',12,4,2,N.array([[1,1,7],[3,5,7]]))
## t=a._1to1_write(1,1,'Connectivity','Zone2',N.array([[3,1,1],[3,5,7]]),N.array([[1,1,1],[1,5,7]]),N.array([1,2,3]))
t=a.conn_write_short(1,1,'Connectivity',3,3,4,2,N.array([[3,1,1],[3,5,7]]),'Zone 02')
print t

a.close()

