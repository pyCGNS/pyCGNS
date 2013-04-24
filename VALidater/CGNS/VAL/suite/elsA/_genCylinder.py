#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.MAP as CGM
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.VAL.simplecheck as CGV
import numpy as NPY
import math

ZonePattern='{Zone-%s}'
BCPattern='{BC-%s}'
C1Pattern='{CT-%s-%s}'

def hasCommonPointRangeAxis(pr1,pr2):
  if (pr1.shape != pr2.shape): return None
  r=[]
  for d in range(len(pr1)):
    r.append((pr1[d][0]==pr2[d][0]) and (pr1[d][1]==pr2[d][1]))
  return tuple(r)

def hasFaceMax(mx,fn,wi):
  imax=mx[0]
  jmax=mx[1]
  kmax=mx[2]
  if    (fn==0): return (tuple(wi.flat)==(   1,imax,   1,   1,   1,kmax))
  elif  (fn==1): return (tuple(wi.flat)==(imax,imax,   1,jmax,   1,kmax))
  elif  (fn==2): return (tuple(wi.flat)==(   1,imax,   1,jmax,kmax,kmax))
  elif  (fn==3): return (tuple(wi.flat)==(   1,   1,   1,jmax,   1,kmax))
  elif  (fn==4): return (tuple(wi.flat)==(   1,imax,   1,jmax,   1,1)   )
  elif  (fn==5): return (tuple(wi.flat)==(   1,imax,jmax,jmax,   1,kmax))
  else:          return False
  
def getMax(Tz):
  return (Tz[1][0][0],Tz[1][1][0],Tz[1][2][0])

def getFaces(Tz):
  (nb,hg,wd)=getMax(Tz)

  bc=[0,0,0,0,0,0]
  bc[0]=NPY.array([[ 1,nb],[ 1, 1],[ 1,wd]],dtype='i',order='F')
  bc[1]=NPY.array([[nb,nb],[ 1,hg],[ 1,wd]],dtype='i',order='F')
  bc[2]=NPY.array([[ 1,nb],[ 1,hg],[wd,wd]],dtype='i',order='F')
  bc[3]=NPY.array([[ 1, 1],[ 1,hg],[ 1,wd]],dtype='i',order='F')
  bc[4]=NPY.array([[ 1,nb],[ 1,hg],[ 1, 1]],dtype='i',order='F')
  bc[5]=NPY.array([[ 1,nb],[hg,hg],[ 1,wd]],dtype='i',order='F')

  return bc

#
# zn: zone tag (name is {Zone-<tag>}, tag is used for BC and Connectivities)
# nb: number of points in the interior/exterior sector (I index)
# ag: angle of the sector (degree)
# sa: starting angle of the sector (degree)
# hg: number of points on the radius (J index)
# wd: number of points on the width (K index)
# rd: radius of outer circle
# rt: radius of inner circle
# wl: width
# sw: starting width
# oc: origin coordinates
#
def genSector(Tb, zn,nb,hg,wd,sa,ag,rd,rt,wl,sw=0,oc=(0,0,0)):
   
  sa=sa*((2*math.pi)/360.)
  ag=ag*((2*math.pi)/360.)

  x=NPY.ones((nb,hg,wd),dtype='d',order='F')
  y=NPY.ones((nb,hg,wd),dtype='d',order='F')
  z=NPY.ones((nb,hg,wd),dtype='d',order='F')

  rds=NPY.linspace(rt,rd,hg,endpoint=True)
  sps=NPY.linspace(sa,ag,nb,endpoint=True)
  wls=NPY.linspace(sw,wl,wd,endpoint=True)

  for i in range(nb):
      for j in range(hg):
          for k in range(wd):
              x[i,j,k]=oc[0]+NPY.cos(sps[i])*rds[j]
              y[i,j,k]=oc[1]+NPY.sin(sps[i])*rds[j]
              z[i,j,k]=oc[2]+wls[k]

  sz=NPY.array([[nb,nb-1,0],[hg,hg-1,0],[wd,wd-1,0]],order='F')

  Tz=CGL.newZone(Tb,ZonePattern%zn,sz)
  Tg=CGL.newGridCoordinates(Tz,CGK.GridCoordinates_s)
  Td=CGL.newDataArray(Tg,CGK.CoordinateX_s,x)
  Td=CGL.newDataArray(Tg,CGK.CoordinateY_s,y)
  Td=CGL.newDataArray(Tg,CGK.CoordinateZ_s,z)

def genBC(Tb,zn,fb,*fn):

  Tz=CGU.hasChildName(Tb,ZonePattern%zn)
  Tg=CGU.hasChildName(Tz,CGK.ZoneBC_s)
  if (Tg is None): Tg=CGL.newZoneBC(Tz)
  
  bc=getFaces(Tz)
  bf=CGK.FamilySpecified_s

  for f in range(6):
    bn=BCPattern%f
    if (f in fn):
        Ta=CGL.newBoundary(Tg,bn,bc[f],btype=bf,family=fb)

def genC1(Tb,zn,*ct):
    
  Tz=CGU.hasChildName(Tb,ZonePattern%zn)
  Tc=CGU.hasChildName(Tz,CGK.ZoneGridConnectivity_s)
  if (Tc is None): Tc=CGL.newZoneGridConnectivity(Tz)

  bc=getFaces(Tz)
  tr=(1,2,3)
  
  for c in ct:
    fn=c[0]
    for zi in c[1:]:
      zd=zi[0]
      Td=CGU.hasChildName(Tb,ZonePattern%(zd))
      bd=getFaces(Td)[zi[1]-1]
      rt=hasFaceMax(getMax(Tz),fn,bc[fn-1])
      print C1Pattern%(zn,zd),bc[fn-1],bd,rt
      if ((len(c[1:])==1) and rt):
          Tq=CGL.newGridConnectivity1to1(Tc,C1Pattern%(zn,zd),ZonePattern%(zd),
                                         bc[fn-1],bd,tr)

if (__name__=='__main__'):
  # Many zones with 1to1 and non-coincident
  R =72
  Ra=24
  Rb=12
  Rc=48
  Rd=36
  W =24
  Wa=18
  Wb=12
  T=CGL.newCGNSTree()
  Tb=CGL.newBase(T,'{Base#1}',3,3)
  Tf=CGL.newFamily(Tb,'Interior')
  Tf=CGL.newFamily(Tb,'Exterior')
  genSector(Tb, 'A',   13,R-Rb+1, 17,    0, 30, R, Rb, W)
  genSector(Tb, 'B',   31,R-Rb+1, 17,   30,120, R, Rb, W)
  genSector(Tb, 'C1',  27,R-Rc+1, 17,  120,200, R, Rc, W)
  genSector(Tb, 'C2',  27,Rc-Rb+1,17,  120,200, Rc,Rb, W)
  genSector(Tb, 'D1',  41,R-Rb+1, 17,  200,360, R, Rb, Wb)
  genSector(Tb, 'D2',  41,R-Rb+1, 17,  200,360, R, Rb, W,Wb)
  genBC(Tb, 'A',  'Interior', 1)
  genBC(Tb, 'A',  'Exterior', 3,5,6)
  genBC(Tb, 'B',  'Interior', 1)
  genBC(Tb, 'B',  'Exterior', 3,5,6)
  genBC(Tb, 'C1', 'Exterior', 3,5,6)
  genBC(Tb, 'C2', 'Interior', 1)
  genBC(Tb, 'C2', 'Exterior', 3,5)
  genBC(Tb, 'D1', 'Interior', 1)
  genBC(Tb, 'D1', 'Exterior', 3,6)
  genBC(Tb, 'D2', 'Interior', 1)
  genBC(Tb, 'D2', 'Exterior', 5,6)
  genC1(Tb, 'A',  (2, ('D1',4,),('D2',4)),(4, ('B',2)))
  genC1(Tb, 'B',  (2,('A',4)),(4,('C1',2),('C2',2)))
  genC1(Tb, 'C1', (1,('C2',6)),(2,('B',4)),(4,('D1',2),('D2',2)))
  genC1(Tb, 'C2', (2,('B',4)),(4,('D1',2),('D2',2)),(6,('C1',1)))
  genC1(Tb, 'D1', (2,('C1',4),('C2',4)),(4,('A',2)),(5,('D2',3)))
  genC1(Tb, 'D2', (2,('C1',4),('C2',4)),(3,('D1',5)),(4,('A',2)))

  if (CGV.compliant(T)): CGM.save('T.hdf',T)
  else: print T
