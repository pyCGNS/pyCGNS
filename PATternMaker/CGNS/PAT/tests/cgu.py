#  ---------------------------------------------------------------------------
#  pyCGNS.PAT - Python package for CFD General Notation System - PATternMaker
#  See license.txt file in the root directory of this Python module source  
#  ---------------------------------------------------------------------------

import unittest

# --------------------------------------------------
# should not be documented with docstrings
class CGUTestCase(unittest.TestCase):
  def setUp(self):
    self.T=None
  def test_00Module(self):
    pass
  def test_02Path(self):
    import CGNS.PAT.cgnsutils as CGU
    p='/Base/Zone/ZoneBC'
    self.assertEqual(p,CGU.getPathNoRoot(p))
    self.assertEqual(CGU.getPathNormalize('/./Base/Zone/ZoneBC'),p)
    self.assertEqual(CGU.getPathToList(p),['','Base', 'Zone', 'ZoneBC'])
    self.assertEqual(CGU.getPathToList(p,True),['Base', 'Zone', 'ZoneBC'])
  def test_01Check(self):
    import CGNS.PAT.cgnsutils  as CGU
    import CGNS.PAT.cgnserrors as CGE
    import numpy as NPY
    name='/name'
    self.assertFalse(CGU.checkNodeName('/name'))
    self.assertFalse(CGU.checkNodeName(3))
    self.assertFalse(CGU.checkNodeName(['name']))
    self.assertFalse(CGU.checkNodeName(''))
    self.assertFalse(CGU.checkNodeName('x'*33))
    self.assertFalse(CGU.checkNodeName('.'))
    self.assertFalse(CGU.checkNodeName('..'))
    self.assertFalse(CGU.checkNodeName('         '))
    self.assertTrue(CGU.checkNodeName('name'))
    self.assertTrue(CGU.checkNodeName('x'*32))
    import string
    clist=list(string.letters+string.digits+string.punctuation+' ')
    clist.remove('/')
    for c in clist:
        self.assertTrue(CGU.checkNodeName('_'+c))
    A=['A',None,[],None]
    B=['B',None,[],None]
    C=['C',None,[],None]
    D=['D',None,[],None]
    A[2]+=[B]
    A[2]+=[C]
    self.assertTrue(CGU.checkDuplicatedName(A,'D'))
    self.assertFalse(CGU.checkHasChildName(A,'D'))
    A[2]+=[D]    
    self.assertFalse(CGU.checkDuplicatedName(A,'D'))
    self.assertTrue(CGU.checkHasChildName(A,'D'))
    self.assertFalse(CGU.checkNodeType(C))
    C[3]='UnknownType_t'
    self.assertFalse(CGU.checkNodeType(C))
    self.assertFalse(CGU.checkNodeType(C,['CGNSBase_t','Zone_t']))
    self.assertFalse(CGU.checkNodeType(C,'CGNSBase_t'))
    C[3]='CGNSBase_t'
    self.assertTrue(CGU.checkNodeType(C))
    self.assertTrue(CGU.checkNodeType(C,['CGNSBase_t','Zone_t']))
    self.assertTrue(CGU.checkNodeType(C,['Zone_t','CGNSBase_t','Zone_t']))
    self.assertTrue(CGU.checkNodeType(C,'CGNSBase_t'))
    
suite = unittest.TestLoader().loadTestsFromTestCase(CGUTestCase)
unittest.TextTestRunner(verbosity=1).run(suite)
