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

def hasFaceMax(Tz,fn,wi):
  (imax,jmax,kmax)=getMax(Tz)
  if    (fn==1): return (tuple(wi.flat)==(   1,imax,   1,jmax,   1,1)   )
  elif  (fn==2): return (tuple(wi.flat)==(   1,imax,   1,   1,   1,kmax))
  elif  (fn==3): return (tuple(wi.flat)==(imax,imax,   1,jmax,   1,kmax))
  elif  (fn==4): return (tuple(wi.flat)==(   1,imax,jmax,jmax,   1,kmax))
  elif  (fn==5): return (tuple(wi.flat)==(   1,   1,   1,jmax,   1,kmax))
  elif  (fn==6): return (tuple(wi.flat)==(   1,imax,   1,jmax,kmax,kmax))
  else:          return False
  
def getMax(Tz):
  return (Tz[1][0][0],Tz[1][1][0],Tz[1][2][0])

def getFaces(Tz):
  (nb,hg,wd)=getMax(Tz)
  bc=[0,0,0,0,0,0]
  bc[0]=NPY.array([[ 1,nb],[ 1,hg],[ 1, 1]],dtype='i',order='F')
  bc[1]=NPY.array([[ 1,nb],[ 1, 1],[ 1,wd]],dtype='i',order='F')
  bc[2]=NPY.array([[nb,nb],[ 1,hg],[ 1,wd]],dtype='i',order='F')
  bc[3]=NPY.array([[ 1,nb],[hg,hg],[ 1,wd]],dtype='i',order='F')
  bc[4]=NPY.array([[ 1, 1],[ 1,hg],[ 1,wd]],dtype='i',order='F')
  bc[5]=NPY.array([[ 1,nb],[ 1,hg],[wd,wd]],dtype='i',order='F')
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
def genSector(Tb, zn,nb,hg,wd,sa,ag,rd,rt,wl,sw=0,oc=(0,0,0),lin=False):
   
  sa=sa*((2*math.pi)/360.)
  ag=ag*((2*math.pi)/360.)

  x =NPY.ones((nb,hg,wd),dtype='d',order='F')
  y =NPY.ones((nb,hg,wd),dtype='d',order='F')
  z =NPY.ones((nb,hg,wd),dtype='d',order='F')

  if lin:
    rds=NPY.linspace(rt,rd,hg,endpoint=True)
  else:
    bs=2
    rds=NPY.logspace(math.log(rt,bs),math.log(rd,bs),hg,endpoint=True,base=bs)
  sps=NPY.linspace(sa,ag,nb,endpoint=True)
  wls=NPY.linspace(sw,wl,wd,endpoint=True)[::-1]

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

  for f in xrange(1,7):
    bn=BCPattern%f
    if (f in fn):
        Ta=CGL.newBoundary(Tg,bn,bc[f-1],btype=bf,family=fb)

def genCT(Tb,zn,*ct):
    
  Tz=CGU.hasChildName(Tb,ZonePattern%zn)
  Tc=CGU.hasChildName(Tz,CGK.ZoneGridConnectivity_s)
  if (Tc is None): Tc=CGL.newZoneGridConnectivity(Tz)

  bc=getFaces(Tz)
  tr=(1,2,3)
  
  for c in ct:
    fn=c[0]
    for zi in c[1:]:
      zd=zi[0]
      # --- 1to1
      if (len(c[1:][0][1:])==1):
        Td=CGU.hasChildName(Tb,ZonePattern%(zd))
        bd=getFaces(Td)[zi[1]-1]
        rt=hasFaceMax(Tz,fn,bc[fn-1])
        if (rt): 
          Tq=CGL.newGridConnectivity1to1(Tc,
                                         C1Pattern%(zn,zd),
                                         ZonePattern%(zd),
                                         bc[fn-1],bd,tr)
      # --- generalized
      else:
        Tq=CGL.newGridConnectivity(Tc,
                                   C1Pattern%(zn,zd),
                                   ZonePattern%(zd),
                                   CGK.Abutting_s)
        if (len(zi)>2):
          CGL.newDataArray(Tq,'ConnectivityKeyLocal',
                           CGU.setStringAsArray(zi[2]))
          CGL.newDataArray(Tq,'ConnectivityKeyDonor',
                           CGU.setStringAsArray(zi[3]))
          CGL.newPointRange(Tq,value=bc[fn-1])
        

def sample():
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
  genSector(Tb, 'C1',  27,R-Rc+1, 17,  120,200, R, Rc, W, lin=True)
  genSector(Tb, 'C2',  27,Rc-Rb+1,17,  120,200, Rc,Rb, W)
  genSector(Tb, 'D1',  41,R-Rb+1, 17,  200,360, R, Rb, Wb)
  genSector(Tb, 'D2',  41,R-Rb+1, 17,  200,360, R, Rb, W,Wb)
  genBC(Tb, 'A',  'Interior', 2)
  genBC(Tb, 'A',  'Exterior', 1,4,6)
  genBC(Tb, 'B',  'Interior', 2)
  genBC(Tb, 'B',  'Exterior', 1,4,6)
  genBC(Tb, 'C1', 'Exterior', 1,4,6)
  genBC(Tb, 'C2', 'Interior', 2)
  genBC(Tb, 'C2', 'Exterior', 1,6)
  genBC(Tb, 'D1', 'Interior', 2)
  genBC(Tb, 'D1', 'Exterior', 4,6)
  genBC(Tb, 'D2', 'Interior', 2)
  genBC(Tb, 'D2', 'Exterior', 1,4)
  genCT(Tb, 'A',  (5, ('D',3,'A_D','D_A')),
                  (3, ('B',5)))
  genCT(Tb, 'B',  (5,('A',3)),
                  (3,('C',5,'B_C','C_B')))
  genCT(Tb, 'C1', (2,('C2',4)),
                  (5,('B',3,'C_B','B_C')),
                  (3,('D',5,'C_D','D_C')))
  genCT(Tb, 'C2', (5,('B',3,'C_B','B_C')),
                  (3,('D',5,'C_D','D_C')),
                  (4,('C1',2)))
  genCT(Tb, 'D1', (5,('C',3,'D_C','C_D')),
                  (3,('A',5,'D_A','A_D')),
                  (1,('D2',6)))
  genCT(Tb, 'D2', (5,('C',3,'D_C','C_D')),
                  (6,('D1',1)),
                  (3,('A',5,'D_A','A_D')))

  return T

#  if (CGV.compliant(T)): CGM.save('T.hdf',T)
#  else: print T
