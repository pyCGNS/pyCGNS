#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll 
import numpy as NPY
import CGNS.PAT.cgnskeywords as CK

print 'CGNS.WRA.mll','#042 - axisym_write'

# ----------------------------------------------------------------------
def acube(im=3,jm=5,km=7,offset=0):
  # inverse k/i in order to get correct order in ADF file
  x=NPY.zeros((km,jm,im),'d')
  y=NPY.zeros((km,jm,im),'d')
  z=NPY.zeros((km,jm,im),'d')
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

a=Mll.pyCGNS('tmp/testmll42.hdf',Mll.MODE_WRITE)
a.base_write('Base',3,2)

a.axisym_write(1,NPY.array([9.8,3]),NPY.array([9.8,3]))

a.close()

# ---
