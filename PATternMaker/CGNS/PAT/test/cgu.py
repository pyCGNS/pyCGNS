#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------
#
import unittest

# --------------------------------------------------
# should not be documented with docstrings
class CGUTestCase(unittest.TestCase):
  def setUp(self):
    self.T=None
  def eStr(self,code):
    import CGNS.PAT.cgnserrors as CGE
    return CGE.TAG+CGE.TAG_ERROR+"\[%.3d\].*$"%code
  def test_00Module(self):
    pass
  def test_01Check(self):
    import CGNS.PAT.cgnsutils  as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    self.assertFalse(CGU.checkName('/name'))
    self.assertFalse(CGU.checkNodeName(3))
    self.assertFalse(CGU.checkName(3))
    self.assertFalse(CGU.checkName(['name']))
    self.assertFalse(CGU.checkName(''))
    self.assertFalse(CGU.checkName('x'*33))
    self.assertFalse(CGU.checkName('.'))
    self.assertFalse(CGU.checkName('..'))
    self.assertFalse(CGU.checkName('         '))
    self.assertTrue(CGU.checkName('name  test'))
    self.assertTrue(CGU.checkName('name"test"'))
    self.assertTrue(CGU.checkName('name\\test'))
    self.assertTrue(CGU.checkName(' name'))
    self.assertTrue(CGU.checkName('name'))
    self.assertTrue(CGU.checkName('x'*32))
    self.assertTrue(CGU.checkNodeName(['test',None,[],'Zone_t']))
    self.assertTrue(CGU.checkNodeName(['  test',None,[],'Zone_t']))
    self.assertTrue(CGU.checkNodeName(['?test#320',None,[],'Zone_t']))
    self.assertTrue(CGU.checkNodeName(['test   ',None,[],'Zone_t']))
    self.assertFalse(CGU.checkName('name  test',strict=True))
    self.assertFalse(CGU.checkName('name\\test',strict=True))
    self.assertFalse(CGU.checkName('name"test"',strict=True))
    self.assertFalse(CGU.checkName(' name',strict=True))
    import string
    clist=list(string.letters+string.digits+string.punctuation+' ')
    clist.remove('/')
    for c in clist:
        self.assertTrue(CGU.checkName('_'+c))
    ex=CGE.cgnsNameError
    fn=CGU.checkName
    self.assertRaisesRegexp(ex,self.eStr(2),CGU.checkNodeName,[],dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(22),fn,3,dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(23),fn,'',dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(24),fn,'/',dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(25),fn,'a'*33,dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(29),fn,'.',dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(31),fn,' '*5,dienow=True)
    self.assertRaisesRegexp(ex,self.eStr(32),fn,' a'*5,dienow=True,strict=True)
    self.assertRaisesRegexp(ex,self.eStr(33),fn,'"',dienow=True,strict=True)
    self.assertRaisesRegexp(ex,self.eStr(34),fn,'a  a',dienow=True,strict=True)
  def test_02Path(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    p='/Base/Zone/ZoneBC'
    self.assertEqual(p,CGU.getPathNoRoot(p))
    self.assertEqual(CGU.getPathNormalize('/./Base/Zone/ZoneBC'),p)
    self.assertEqual(CGU.getPathNormalize('/./Base/Zone/ZoneBC//'),p)
    self.assertEqual(CGU.getPathToList(p),['','Base', 'Zone', 'ZoneBC'])
    self.assertEqual(CGU.getPathToList(p,True),['Base', 'Zone', 'ZoneBC'])
  def test_03NodeStructure(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    self.assertEqual(['N1',None,[],'N_t'],CGU.nodeCreate('N1',None,[],'N_t'))
    p=CGU.nodeCreate('N1',None,[],'N_t')
    self.assertEqual(['N1',None,[],'N_t'],CGU.nodeCreate('N1',None,[],'N_t',p))
    self.assertEqual(['N1',None,[],'N_t'],p[2][0])
    n=CGU.nodeCreate('N2',None,[],'N_t',p)
    self.assertNotEqual(['N1',None,[],'N_t'],p[2][1])
    self.assertIs(n,p[2][1])
    self.assertEqual(['N2',None,[],'N_t'],CGU.nodeCopy(n))
    c=CGU.nodeCopy(n)
    self.assertIsNot(n,c)
    cname='NCopy'
    c=CGU.nodeCopy(n,cname)    
    self.assertEqual(c[0],cname)
    del n
    self.assertEqual(c[0],cname)
  def test_04NodeChildren(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    A=['A',None,[],None]
    B=['B',None,[],None]
    C=['C',None,[],None]
    D=['D',None,[],None]
    A[2]+=[B]
    A[2]+=[C]
    self.assertTrue(CGU.hasChildName(A,'B'))
    self.assertTrue(CGU.checkDuplicatedName(A,'D'))
    self.assertFalse(CGU.checkHasChildName(A,'D'))
    A[2]+=[D]    
    self.assertFalse(CGU.checkDuplicatedName(A,'D'))
    self.assertTrue(CGU.checkHasChildName(A,'D'))
    self.assertFalse(CGU.checkNodeType(C))
  def test_05NodeValue(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    pass
  def test_06NodeType(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    A=['A',None,[],None]
    B=['B',None,[],None]
    C=['C',None,[],None]
    D=['D',None,[],None]
    A[2]+=[B]
    A[2]+=[C]
    C[3]='UnknownType_t'
    self.assertFalse(CGU.checkNodeType(C))
    self.assertFalse(CGU.checkNodeType(C,[CGK.CGNSBase_ts,CGK.Zone_ts]))
    self.assertFalse(CGU.checkNodeType(C,CGK.CGNSBase_ts))
    C[3]=CGK.CGNSBase_ts
    self.assertTrue(CGU.checkNodeType(C))
    self.assertTrue(CGU.checkNodeType(C,[CGK.CGNSBase_ts,CGK.Zone_ts]))
    self.assertTrue(CGU.checkNodeType(C,[CGK.Zone_ts,CGK.CGNSBase_ts,
                                         CGK.Zone_ts]))
    self.assertTrue(CGU.checkNodeType(C,CGK.CGNSBase_ts))
  def test_07NodeCompliance(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    import numpy
    n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
    z=CGU.nodeCreate('ReferenceState',None,[],CGK.ReferenceState_ts,parent=n)
    self.assertTrue(CGU.checkNodeCompliant(n,None))
    self.assertTrue(CGU.checkNodeCompliant(z,None))
    self.assertTrue(CGU.checkNodeCompliant(z,n,dienow=True))
    self.assertFalse(CGU.checkNodeCompliant(None,n))
  def test_08NodeRetrieval(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    pass
  def test_09NodeDelete(self):
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnserrors as CGE
    import CGNS.PAT.cgnskeywords as CGK
    import numpy
    n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
    r=CGU.nodeCreate('ReferenceState',None,[],CGK.ReferenceState_ts,parent=n)
    d=CGU.nodeCreate('Data',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)
    self.assertIsNotNone(CGU.hasChildName(r,'Data'))
    CGU.nodeDelete(n,d)
    self.assertIsNone(CGU.hasChildName(r,'Data'))                    
    d=CGU.nodeCreate('DataZ',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)
    self.assertIsNotNone(CGU.hasChildName(r,'DataZ'))
    CGU.nodeDelete(r,d)
    self.assertIsNone(CGU.hasChildName(r,'DataZ'))                    
    CGU.nodeDelete(r,d)
    self.assertIsNone(CGU.hasChildName(r,'DataZ'))                    
    n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
    r=CGU.nodeCreate('ReferenceState',None,[],CGK.ReferenceState_ts,parent=n)
    d=CGU.nodeCreate('Data',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)
    self.assertIsNotNone(CGU.getNodeByPath(n,'/Base/ReferenceState/Data'))
    CGU.nodeDelete(n,d)
    self.assertIsNone(CGU.hasChildName(r,'Data'))              
    n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
    r=CGU.nodeCreate('ReferenceState',None,[],CGK.ReferenceState_ts,parent=n)
    d=CGU.nodeCreate('Data',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)
    self.assertIsNotNone(CGU.getNodeByPath(n,'/Base/ReferenceState/Data'))
    CGU.removeChildByName(r,'Data')
    self.assertIsNone(CGU.hasChildName(r,'Data'))              

# ---
suite = unittest.TestLoader().loadTestsFromTestCase(CGUTestCase)
unittest.TextTestRunner(verbosity=1).run(suite)
# ---
