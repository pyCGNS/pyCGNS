
import CGNS.PAT.cgnslib      as CGL
import CGNS.PAT.cgnsutils    as CGU
import CGNS.PAT.cgnskeywords as CGK
import CGNS.MAP              as CGM
import CGNS.VAL.simplecheck  as CGV

import numpy as NPY

def mergeTrees(treelist):
    """List order is meaningful: the first tree is set as the master tree,
    if a conflict occurs and the FORCEFIRST flag is on, the master tree
    overwrites the slave tree...
    There is no loop detection"""
    if (len(treelist)<2): return treelist[0]
    tmaster=treelist[0]
    for tslave in treelist[1:]:
        tmaster=mergeTwoTrees(tmaster,tslave)
    return tmaster

def mergeTwoTrees(TM,TS,path=''):
    TR=[TM[0],TM[1],[],TM[3]]
    C_M=set(CGU.childrenNames(TM))
    C_S=set(CGU.childrenNames(TS))
    C_L=C_M.symmetric_difference(C_S)
    for C in C_L:
        #print 'add   ',path+'/'+C
        if (C in C_M): T=TM
        else:          T=TS
        C_N=CGU.hasChildNode(T,C)
        TR[2].append(C_N)
    C_L=C_M.intersection(C_S)
    for C in C_L:
        #print 'merge ',path+'/'+C
        TR[2].append(mergeTwoTrees(CGU.hasChildNode(TM,C),
                                   CGU.hasChildNode(TS,C),
                                   path+'/'+C))
    return TR

def test():
    T1=CGL.newCGNSTree()
    b1=CGL.newCGNSBase(T1,'B1',3,3)
    b2=CGL.newCGNSBase(T1,'B2',3,3)
    f1=CGL.newFamily(b1,'F1')
    f2=CGL.newFamily(b1,'F2')
    f1=CGL.newFamily(b2,'F1')
    f2=CGL.newFamily(b2,'F2')
    f3=CGL.newFamily(b2,'F3')
    f4=CGL.newFamily(b2,'F4')
    z1=CGL.newZone(b1,'Z1',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z2=CGL.newZone(b1,'Z2',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z3=CGL.newZone(b1,'Z3',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z4=CGL.newZone(b1,'Z4',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z5=CGL.newZone(b1,'Z5',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z1=CGL.newZone(b2,'Z1',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z2=CGL.newZone(b2,'Z2',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z3=CGL.newZone(b2,'Z3',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))

    T2=CGL.newCGNSTree()
    b1=CGL.newCGNSBase(T2,'B1',3,3)
    b2=CGL.newCGNSBase(T2,'B2',3,3)
    b3=CGL.newCGNSBase(T2,'B3',2,2)
    f1=CGL.newFamily(b1,'F1')
    f1=CGL.newFamily(b2,'F1')
    f2=CGL.newFamily(b2,'F2')
    f3=CGL.newFamily(b2,'F3')
    f4=CGL.newFamily(b2,'F4')
    f5=CGL.newFamily(b2,'F5')
    f1=CGL.newFamily(b3,'F1')
    f2=CGL.newFamily(b3,'F2')
    z1=CGL.newZone(b1,'Z1',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z2=CGL.newZone(b1,'Z2',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z3=CGL.newZone(b1,'Z3',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z1=CGL.newZone(b2,'Z1',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z2=CGL.newZone(b2,'Z2',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z3=CGL.newZone(b2,'Z3',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))
    z4=CGL.newZone(b2,'Z4',NPY.array([[5,4,0],[7,6,0],[9,8,0]],order='F'))

    T3=CGL.newCGNSTree()
    b1=CGL.newCGNSBase(T3,'B1',3,3)
    b2=CGL.newCGNSBase(T3,'B7',3,3)
    b3=CGL.newCGNSBase(T3,'B8',2,2)
    f1=CGL.newFamily(b1,'F1')
    f1=CGL.newFamily(b2,'F8')

    Tlist=[]
    for TT in [T1,T2,T3]: Tlist.append(TT)

    count=0
    for t in Tlist:
        CGM.save('merge-T%.2d.hdf'%count,t,flags=CGM.S2P_TRACE)
        count+=1
    T=mergeTrees(Tlist)
    CGM.save('merge-Result.hdf',T,flags=CGM.S2P_TRACE)
    return T
