#  -------------------------------------------------------------------------
#  pyCGNS.VAL - Python package for CFD General Notation System - VALidater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
import CGNS.MAP as CGM
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import numpy as NPY
import math
#
# zn: zone name
# nb: number of points in the interior/exterior sector (I index)
# ag: angle of the sector (degree)
# sa: starting angle of the sector (degree)
# hg: number of points on the radius (J index)
# wd: number of points on the width (K index)
# rd: radius of outer circle
# rt: radius of inner circle
# wl: width
# oc: origin coordinates
#
def genSector(zn,nb,hg,wd,sa,ag,rd,rt,wl,oc):
    
   sa=sa*((2*math.pi)/360.)
   ag=ag*((2*math.pi)/360.)

   x=NPY.ones((nb,hg,wd),dtype='d',order='F')
   y=NPY.ones((nb,hg,wd),dtype='d',order='F')
   z=NPY.ones((nb,hg,wd),dtype='d',order='F')

   rds=NPY.linspace(rt,rd,hg,endpoint=True)
   sps=NPY.linspace(sa,ag,nb,endpoint=True)
   wls=NPY.linspace(0,wl,wd,endpoint=True)

   for i in range(nb):
       for j in range(hg):
           for k in range(wd):
               x[i,j,k]=oc[0]+NPY.cos(sps[i])*rds[j]
               y[i,j,k]=oc[1]+NPY.sin(sps[i])*rds[j]
               z[i,j,k]=oc[2]+wls[k]

   sz=NPY.array([[nb,nb-1,0],[hg,hg-1,0],[wd,wd-1,0]],order='F')

   Tz=CGL.newZone(None,zn,sz)
   Tg=CGL.newGridCoordinates(Tz,'{Grid#001}')
   Td=CGL.newDataArray(Tg,CGK.CoordinateX_s,x)
   Td=CGL.newDataArray(Tg,CGK.CoordinateY_s,y)
   Td=CGL.newDataArray(Tg,CGK.CoordinateZ_s,z)

   return Tz

T=CGL.newCGNSTree()
Tb=CGL.newBase(T,'{Base#1}',3,3)
Tz=genSector('{Zone-A}',27,19,17,0,30,1,0.2,0.3,(0,0,0))
CGU.addChild(Tb,Tz)
Tz=genSector('{Zone-B}',27,19,17,30,190,1,0.2,0.3,(0,0,0))
CGU.addChild(Tb,Tz)
Tz=genSector('{Zone-C}',27,19,17,190,270,1,0.2,0.3,(0,0,0))
CGU.addChild(Tb,Tz)
Tz=genSector('{Zone-D}',27,19,17,270,360,1,0.2,0.3,(0,0,0))
CGU.addChild(Tb,Tz)
CGM.save('T.hdf',T)
