#  ---------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  ---------------------------------------------------------------------------
#
from __future__ import unicode_literals
from __future__ import print_function
from functools import cmp_to_key
from builtins import (str, bytes, range, dict)

import hashlib
import os.path as op
import re
import string

import numpy

import CGNS
import CGNS.PAT.cgnserrors as CE
import CGNS.PAT.cgnskeywords as CK
import CGNS.PAT.cgnstypes as CT

import sys
PY3 = sys.version_info[0] == 3
# from os import sep as SEPARATOR

__I4 = numpy.dtype(numpy.int32)
__I8 = numpy.dtype(numpy.int64)
__R4 = numpy.dtype(numpy.float32)
__R8 = numpy.dtype(numpy.float64)


# -----------------------------------------------------------------------------
# undocumented functions are private (or obsolete)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def nodeCreate(name, value, children, ntype, parent=None, dienow=False):
    """
    Create a new node with and bind it to its parent::

      import CGNS.PAT.cgnskeywords as CK
      import numpy

      n=nodeCreate('Base',numpy.array([3,3]),[],CK.CGNSBase_ts)
      z=nodeCreate('ReferenceState',None,[],CK.ReferenceState_ts,parent=n)

      # you can safely create a node without referencing its return,
      # the actual new node is stored into the parent's children list
      nodeCreate('ReferenceState',None,[],CK.ReferenceState_ts,parent=n)

    :arg str name: new node name
    :arg numpy.ndarray value: node value
    :arg list children: list of node children (list of CGNS/Python nodes)
    :arg str ntype: CGNS/SIDS node type
    :arg node parent: node where to insert the new node (default: None)
    :arg bool dienow: If `True` raises an exception on error (default:`False`)

    :return: the new node

    :Remarks:
      - If parent is None (default) node is orphan (see comment in example)
      - Adds checks with :py:func:`checkNodeCompliant` only if `dienow` is `True`

    """
    return newNode(name, value, children, ntype, parent=parent, dienow=dienow)


def newNode(name, value, children, ntype, parent=None, dienow=False):
    node = [name, value, [], ntype]
    if isinstance(children, list) and children is not []:
        if checkNodeCompliant(children):
            setAsChild(node, children)
        else:
            node[2] = children
    if dienow:
        checkNodeCompliant(node, parent, dienow)
    if parent:
        setAsChild(parent, node)
    return node


# --------------------------------------------------
def nodeCopy(node, newname=None, share=False):
    """
    Creates a new node sub-tree as a copy of the argument node sub-tree.
    A deep copy is performed on the node, including the values, which can
    lead to a very large memory use::

      n1=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity')
      n2=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity/Connect1')
      n3=nodeCopy(n2,'Connect2')
      nodeChild(n1,n3)

    :arg node node: node to copy
    :arg str newname: new node (copy) name
    :arg bool share: if True then actual numpy.ndarray in not copied (if possible)

    :return: The new node

    :Remarks:
      - The new node name is the same by default, thus user would have to check
        for potential duplicated name
      - The node value is a copy too, numpy.ndarray is duplicated
      - Default is to deep copy including numpy.ndarray
    """
    if newname is None:
        newname = node[0]
    return copyNode(node, newname, share)


def copyNode(n, newname, share=False):
    if not share:
        newn = [newname, copyArray(n[1]), deepcopyNodeList(n[2], share=share), n[3]]
    else:
        newn = [newname, n[1], deepcopyNodeList(n[2], share=share), n[3]]
    return newn


# --------------------------------------------------
def nodeDelete(tree, node, legacy=False):
    """
    Deletes a node from a tree::

      import CGNS.PAT.cgnslib as CL
      import CGNS.PAT.cgnsutils as CU
      import numpy

      T =CL.newCGNSTree()
      b1=CL.newBase(T,'Base',3,3)
      z1=CL.newZone(b1,'Zone1',numpy.array([1,2,3]))
      z2=CL.newZone(b1,'Zone2',numpy.array([1,2,3]))
      print CU.getPathFullTree(T)
      # ['/CGNSLibraryVersion', '/Base',
      #  '/Base/Zone1','/Base/Zone1/ZoneType',
      #  '/Base/Zone2', '/Base/Zone2/ZoneType']
      CU.nodeDelete(T,z1)
      print CU.getPathFullTree(T)
      # ['/CGNSLibraryVersion', '/Base', '/Base/Zone2', '/Base/Zone2/ZoneType']

    :arg node tree: target tree where to find the node to remove
    :arg node: a CGNS/Python node or str (node name) as absolute path to remove

    :Return: The tree argument (without the deleted node)

    :Remarks:
      - Uses :py:func:`checkSameNode`.
      - Actual memory of the node is released only if there is no other reference
      - No action if the node to delete is not found
    """
    if isinstance(node, str):
        path = node
    else:
        path = getPathFromRoot(tree, node)
    if path is not None:
        pp = getPathAncestor(path)
        pc = getPathLeaf(path)
        if pp != path:
            np = getNodeByPath(tree, pp)
            if np is not None:
                removeChildByName(np, pc)
    return tree


# --------------------------------------------------
def deepcopyNodeList(la, share=False):
    if not la:
        return la
    ra = []
    for a in la:
        ra.append(copyNode(a, a[0], share))
    return ra


# -----------------------------------------------------------------------------
# support functions
# -----------------------------------------------------------------------------
def checkNodeName(node, dienow=False):
    """
    Checks if the name is CGNS/Python compliant node name::

      if (not checkNodeName(node)):
          print 'Such name ',name,' not allowed'

    :arg CGNS/Python node: node to check
    :return: True if the name has a correct syntax
    :Remarks:
      - The function is the same as :py:func:`checkName` but with a node as
        arg instead of string
      - See also :py:func:`checkNodeCompliant`
    """
    if not checkNode(node):
        if dienow:
            raise CE.cgnsNameError(2)
        return False
    return checkName(node[0], dienow)


def checkName(name, dienow=False, strict=False):
    """
    Checks if the name is CGNS/Python compliant node name. The name should be a
    Python string of type ``str`` or ``unicode``, but ``str`` is strongly
    recommended.
    A name should follow these requirements:

    * No more than 32 chars
    * No ``/`` in the string
    * Empty name or name with only spaces is forbidden
    * Names ``.`` and ``..`` are forbidden
    * Allowed chars are :py:data:`string.ascii_letters` + :py:data:`string.digits` + :py:data:`string.punctuation` + ``' '``

    Additional checks can be performed with the `strict` flag set to True, these
    checks are not CGNS compliant:

    * No prefix nor suffix spaces
    * No more than two consecutive spaces
    * Punctuation is limited to:  ``!#$%&()*+,-.:;<=>?@[]^_{|}~``

    The check following pattern shows a two steps verification::

       if (not checkName(name)):
          raise Exception('Such name %s not allowed'%name)
       elif (not checkName(name,strict=True)):
          print 'name ok but unsafe'

    :arg str name: string to check
    :arg bool strict: forces the 'strict' mode checks (default is False)
    :return: True if the name has a correct syntax
    :Remarks:
      - Type of name should be a str
      - Name cannot be empty
      - There is no way to check against a regular expression

    :raises: codes 22,23,24,25,29,31,32,33,34 if `dienow` is True
    """
    if not isinstance(name, str):
        if dienow:
            raise CE.cgnsNameError(22)
        return False
    if len(name) == 0:
        if dienow:
            raise CE.cgnsNameError(23)
        return False
    sname = set(name)
    rname = set(string.digits
                + string.ascii_letters
                + string.punctuation + ' ')
    rname.remove('/')
    if not sname.issubset(rname):
        if dienow:
            raise CE.cgnsNameError(24, name)
        return False
    if len(name) > 32:
        if dienow:
            raise CE.cgnsNameError(25, name)
        return False
    if name in ['.', '..']:
        if dienow:
            raise CE.cgnsNameError(29)
        return False
    if name.count(' ') == len(name):
        if dienow:
            raise CE.cgnsNameError(31)
        return False
    if strict:
        if (name.lstrip() != name) or (name.rstrip() != name):
            if dienow:
                raise CE.cgnsNameError(32)
            return False
        rname.remove('"')
        rname.remove('\\')
        rname.remove("'")
        rname.remove("`")
        if not sname.issubset(rname):
            if dienow:
                raise CE.cgnsNameError(33)
            return False
        if '  ' in name:
            if dienow:
                raise CE.cgnsNameError(34)
            return False
    return True


def checkNameOrGenerate(name, pattern='%32s'):
    """
    Checks if a name is CGNS/SIDS compliant, if not generate a new
    name using a hash fonction with arg name as input.
    The check is performed using checkName.
    A pattern can be used only if it actually leads to a compliant name::

      name=checkNameOrGenerate(name,pattern='GEN:%s'
    """
    if checkName(name, strict=True):
        return name
    h = hashlib.md5()
    h.update(name)
    nname = h.hexdigest()
    name = '%32s' % (pattern % nname)[:32]
    if checkName(name, strict=True):
        return name
    return nname


# --------------------------------------------------
def addChild(parent, node):
    return setChild(parent, node)


def setAsChild(parent, node):
    """
    Adds a child node to the parent node children list::

      n1=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity')
      n2=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity/Connect1')
      n3=nodeCopy(n2)
      setChild(n1,n3)

    :arg CGNS/Python parent: the parent node
    :arg CGNS/Python node: the child node to add to parent
    :return: The parent node
    :Remarks:
      - No check is performed on duplicated child name or any other validity.

    """
    return setChild(parent, node)


def setAsUniqueChild(parent, node, check=False):
    """
    Adds a child node to the parent node children list
    Existing child node with the same name is replaced::

      n1=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity')
      n2=getNodeByPath(T,'/Base/Zone1/ZoneGridConnectivity/Connect1')
      n3=nodeCopy(n2)
      setAsUniqueChild(n1,n3)

    :arg CGNS/Python parent: the parent node
    :arg CGNS/Python node: the child node to add to parent
    :return: The parent node
    :Remarks:
      - Check is performed on duplicated child name.
      - Check on node is optional

    """
    if check:
        valid = (checkNode(node)
                 and checkName(node[0])
                 and checkArray(node[1])
                 and checkNodeType(node))
        if not valid:
            return parent
    for idx, child in enumerate(parent[2]):
        if child[0] == node[0]:
            parent[2][idx] = node
            break
    else:
        parent[2].append(node)
    return parent


def setChild(parent, node):
    checkNodeCompliant(node, parent)
    parent[2].append(node)
    return parent


# -----------------------------------------------------------------------------
def checkDuplicatedName(parent, name, dienow=False):
    """
    Checks if the name is not already in the children list of the parent::

      count=1
      while (not checkDuplicatedName(node,'solution#%.3d'%count)): count+=1

    :arg CGNS/Python parent: the parent node
    :arg str name: the child name to look for
    :return:
      - True if the child *IS NOT* duplicated
      - False if the child *IS* duplicated
    :Remarks:
      - Bad legacy interface, True means not ok (see :py:func:`checkChildName`)
    :raise: :ref:`cgnsnameerror` code 102 if `dienow` is True
    """
    if not parent:
        return True
    if parent[2] is None:
        return True
    for nc in parent[2]:
        if nc[0] == name:
            if dienow:
                raise CE.cgnsNameError(102, (name, parent[0]))
            return False
    return True


# -----------------------------------------------------------------------------
def checkHasChildName(parent, name, dienow=False):
    return checkChildName(parent, name, dienow)


# -----------------------------------------------------------------------------
def checkChildName(parent, name, dienow=False):
    """
    Checks if the name is in the children list of the parent::

      count=1
      while (checkChildName(node,'solution#%.3d'%count)): count+=1

    :arg CGNS/Python parent: the parent node
    :arg str name: the child name to look for

    :return: True if the child exists

    :Remarks:

      - Same function as ``checkDuplicatedName`` but with the **NOT** return
        (see :py:func:`checkDuplicatedName`)

    :raise: :ref:`cgnsnameerror` code 102 if `dienow` is True
    """
    return not checkDuplicatedName(parent, name, dienow)


# -----------------------------------------------------------------------------
def checkUniqueChildName(parent, name, dienow=False):
    """
    Checks if the name is unique in the children list of the parent::

    :arg CGNS/Python parent: the parent node
    :arg str name: the child name to look for
    :return:
      - True if the child *IS* Unique or Not Used
      - False if the child *IS NOT* Unique

    :raise: :ref:`cgnsnameerror` code 102 if `dienow` is True
    """
    if not parent:
        return True
    if parent[2] is None:
        return True
    count = sum(1 for child in parent[2] if child[0] == name)
    status = count < 2
    if dienow and not status:
        raise CE.cgnsNameError(102, (name, parent[0]))
    return status


# -----------------------------------------------------------------------------
def getChildName(parent, name, dienow=False):
    """
    Checks if the name is in the children list of the parent, if the name
    already exists, a new name is returned. If the name doesn't exist the
    name itself is returned::

    z=CGU.setAsChild(T,CGU.getChildName(T,'BASE'))

    :arg CGNS/Python parent: the parent node
    :arg str name: the child name to look for
    :return:
      - arg name is it doesn't exist
      - a new unique name if arg name already exists
    """
    if checkDuplicatedName(parent, name, dienow):
        return name
    count = 1
    while checkChildName(parent, '%s#%.d' % (name, count)):
        count += 1
    return '%s#%.d' % (name, count)


# -----------------------------------------------------------------------------
def checkNodeType(node, cgnstype=None, dienow=False):
    """
    Check the CGNS type of a node. The type can be a single value or
    a list of values. Each type to check is a string such as
   `CGNS.PAT.cgnskeywords.CGNSBase_ts` constant for example.
    If the list is empty, the check uses the list of all existing CGNS types::

      import CGNS.PAT.cgnskeywords as CK

      n=nodeCreate('Base',numpy([3,3]),[],CK.CGNSBase_ts)
      checkNodeType(n)
      checkNodeType(n,['Zone_t',CK.CGNSBase_ts])
      checkNodeType(n,CGK.FamilyName_ts)

    :arg node node: target node to check
    :arg list cgnstype: a list of strings with the CGNS/SIDS types to check

    :return:
     * `True` if the type is a CGNS/SIDS the argument list.
     * `False` if the type is a CGNSType or a type in the argument list.
     * `None` if the parent is None (`may change to have consistent return`)

    :raises: :ref:`cgnstypeerror` codes 103,104,40 if `dienow` is True

    :Remarks:
       - The default value for `cgnstype` is the list of all CGNS/SIDS types
    """
    if cgnstype is None:
        cgnstype = []
    return checkType(node, cgnstype, '', dienow)


def checkType(parent, ltype, name, dienow=False):
    """same as :py:func:`checkNodeType <CGNS.PAT.cgnsutils.checkNodeType>`"""
    if parent is None:
        return None
    if (ltype == []) and (parent[3] in CK.cgnstypes):
        return True
    if isinstance(ltype, list) and (parent[3] in ltype):
        return True
    if parent[3] == ltype:
        return True
    if (ltype == []) and (parent[3] not in CK.cgnstypes):
        if dienow:
            raise CE.cgnsTypeError(40, (parent, parent[3]))
        return False
    if parent[3] != ltype:
        if dienow:
            raise CE.cgnsTypeError(103, (parent, ltype))
        return False
    if isinstance(ltype, list) and (parent[3] not in ltype):
        if dienow:
            raise CE.cgnsTypeError(104, (parent, ltype))
        return False
    return True


# -----------------------------------------------------------------------------
def checkParentType(parent, stype):
    """same as :py:func:`checkType <CGNS.PAT.cgnsutils.checkType>` but
     checks the parent type instead of the current node"""
    if parent is None:
        return False
    if parent[3] != stype:
        return False
    return True


# -----------------------------------------------------------------------------
def checkTypeList(parent, ltype, name):
    if parent is None:
        return None
    if parent[3] not in ltype:
        raise CE.cgnsException(104, (name, ltype))
    return None


# -----------------------------------------------------------------------------
def checkParent(node, dienow=False):
    if node is None:
        return False
    return checkNode(node, dienow)


# -----------------------------------------------------------------------------
def checkNode(node, dienow=False):
    """
    Checks if a node is a compliant CGNS/Python node structure of list.
    The syntax for a compliant node structure is:

    .. code-block:: c

      [<name:string>, <value:numpy>, <children:list-of-nodes>, <cgnstype:string>]

    With the following exception: a `value` can be None.

    The function checks the syntax of the node and the types of its contents,
    it doesn't perform sub checks such as `checkNodeName`, `checkNodeType`...

    You should always check first the node structure, a function such as
    `checkNodeName` blindly access to the first item of the list and would raise
    an exception if the structure is not correct.

    :arg CGNS/Python node: the CGNS/Python node to check
    :return: True if the node is ok
    :Remarks:
      - see also :py:func:`checkNodeCompliant`
    :raise: :ref:`cgnsnodeerror` codes 1,2,3,4,5 if `dienow` is True
    """
    if not node:
        if dienow:
            raise CE.cgnsException(1)
        return False
    if not isinstance(node, list):
        if dienow:
            raise CE.cgnsException(2)
        return False
    if len(node) != 4:
        if dienow:
            raise CE.cgnsException(2)
        return False
    if not isinstance(node[0], str):
        if dienow:
            raise CE.cgnsException(3)
        return False
    if not isinstance(node[2], list):
        if dienow:
            raise CE.cgnsException(4, node[0])
        return False
    if (node[1] is not None) and (not isinstance(node[1], numpy.ndarray)):
        if dienow:
            raise CE.cgnsException(5, node[0])
        return False
    return True


# -----------------------------------------------------------------------------
def checkRootNode(node, legacy=False, dienow=False):
    """
    Checks if a node is the CGNS/Python tree root node.
    If `legacy` is True, then `[None, None, [children], None]` is
    accepted as Root. If it is not True (default) then a CGNS/Python node
    of type `CGNSTree_t` is expected as root node.
    Children contains then the `CGNSLibraryVersion`
    and `CGNSBase` nodes as flat list.
    The `flat` pattern with a list containing `CGNSLibraryVersion`
    and zero or more `CGNSBase` nodes is not accepted. You should use a trick as
    the one above by giving this pattern as child of a fake node::

       # flatpattern is a CGNS/Python list with a `CGNSLibraryVersion` node
       # and `CGNSBase` nodes at the same level.
       # we build a temporary children list

       tmp=[None, None, [flatpattern], None]
       if (checkRootNode(tmp,True)):
           print 'CGNS/Python tree has a legacy root node'

    :arg CGNS/Python node: the node to check
    :arg bool legacy: True means you accept non-CGNSTree_t node
    :return: True if the node is a root node
    :raises: :ref:`cgnsnodeerror` codes 90,91,99 if `dienow` is True
    """
    return isRootNode(node, legacy, dienow)


def isRootNode(node, legacy=False, version=False, dienow=False):
    if not node:
        if dienow:
            raise CE.cgnsNodeError(90)
        return False

    if not checkNode(node, dienow):
        return False
    if legacy:
        start = node
    else:
        if node[3] == CK.CGNSTree_ts:
            start = node
        else:
            return False
    versionfound = 0
    cgnsversion = 0.0
    for n in start[2]:
        if not checkNode(n, dienow):
            return False
        if ((n[0] == CK.CGNSLibraryVersion_s) and
                (n[3] == CK.CGNSLibraryVersion_ts)):
            if versionfound and dienow:
                raise CE.cgnsNodeError(99)
            if versionfound != 0:
                return False
            versionfound = 1
            if version and n[1] is not None:
                cgnsversion = n[1][0]
        elif n[3] != CK.CGNSBase_ts:
            if dienow:
                raise CE.cgnsNodeError(91)
            return False
    if version:
        return cgnsversion
    if versionfound:
        return True
    else:
        return True


def getVersion(tree):
    v = isRootNode(tree, version=True)
    if v:
        return v
    return 0.0


# -----------------------------------------------------------------------------
# Arbitrary and incomplete node comparison (lazy evaluation)
def checkSameTree(treeA, treeB, dienow=False):
    """
    Checks if two trees are the same, recursive use of CheckSameNode
    Does not says where it may differ, only True or False is returned
    """
    try:
        checkSameNode(treeA, treeB, dienow=True)
        for a, b in zip(getNextChildSortByType(treeA), getNextChildSortByType(treeB)):
            checkSameNode(a, b, dienow=True)
    except CE.cgnsNodeError:
        return False
    return True


# -----------------------------------------------------------------------------
# Arbitrary and incomplete node comparison (lazy evaluation)
def checkSameNode(nodeA, nodeB, dienow=False):
    """
    Checks if two nodes have the same contents: same name, same CGNS/SIDS type,
    same number of children, same value type (None of numpy).
    The children are not parsed, the value itself is not checked
    (see :py:func:`checkSameValue`)::

      if checkSameNode(nodeA,nodeB):
          nodeB = copyNode(nodeB)

    :arg CGNS/Python nodeA: first node to compare
    :arg CGNS/Python nodeB: second node to compare
    :return: True if the nodes are same
    :remarks:
      - the function call :py:func:`checkSameValue`
    :raise: :ref:`cgnsnodeerror` code 30 if `dienow` is True
    """
    return sameNode(nodeA, nodeB, dienow)


def sameNode(nodeA, nodeB, dienow=False):
    same = True
    if not (checkNode(nodeA) and checkNode(nodeB)):
        same = False
    elif nodeA[0] != nodeB[0]:
        same = False
    elif nodeA[3] != nodeB[3]:
        same = False
    elif not isinstance(nodeA[1], type(nodeB[1])):
        same = False
    elif not checkSameValue(nodeA, nodeB):
        same = False
    elif len(nodeA[2]) != len(nodeB[2]):
        same = False
    if not same and dienow:
        raise CE.cgnsNodeError(30, (nodeA[0], nodeB[0]))
    return same


# -----------------------------------------------------------------------------
def checkSameValue(nodeA, nodeB, dienow=False):
    """
    Checks if two nodes have the same value. There is no tolerance on actual
    array values when these are compared one to one. This could lead to time
    consuming operations for large arrays::

       if not checkSameValue(nodeA,nodeB):
           raise Exception('Not the same value')

    :arg CGNS/Python nodeA: first node to compare the value with
    :arg CGNS/Python nodeB: second node to compare the value with
    :return: True if the nodes have same value
    :raise: :ref:`cgnsnodeerror` code 30 if `dienow` is True
    """
    v_a = nodeA[1]
    v_b = nodeB[1]
    if (v_a is None) and (v_b is None):
        return True
    if (v_a is None) or (v_b is None):
        return False
    if isinstance(v_a, numpy.ndarray) and isinstance(v_b, numpy.ndarray):
        if v_a.dtype != v_b.dtype:
            return False
        if v_a.shape != v_b.shape:
            return False
        return numpy.all(numpy.equal(v_a, v_b))
    if isinstance(v_a, numpy.ndarray) or isinstance(v_b, numpy.ndarray):
        return False
    return v_a == v_b


# -----------------------------------------------------------------------------
def checkArray(a, dienow=False, orNone=True):
    """
    Check if the array value of a node is a numpy array::

       if not checkArray(node[1]):
           raise Exception('Bad array (for a CGNS/Python node)')

    :arg numpy.ndarray a: value to check
    :arg bool dienow: If `True` raises an exception on error (default:`False`)
    :arg bool orNone: If `True' return True when `a' is None
    :return: True if the arg array is suitable as CGNS/Python value
    :raise: error codes 109,170 if `dienow` is True
    """
    if orNone and (a is None):
        return True
    if not isinstance(a, numpy.ndarray):
        if dienow:
            raise CE.cgnsException(109)
        return False
    if getValueType(a) is None:
        if dienow:
            raise CE.cgnsException(111)
        return False
    if len(a.shape) > 1 and not a.flags.f_contiguous:
        if dienow:
            raise CE.cgnsException(710)
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayChar(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type C1"""
    if not checkArray(a, dienow):
        return False
    if a.dtype.kind not in ['S', 'a']:
        if dienow:
            raise CE.cgnsException(105)
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayI4(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type I4"""
    if not checkArray(a):
        return False
    if a.dtype != __I4:
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayI8(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type I8"""
    if not checkArray(a):
        return False
    if a.dtype != __I8:
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayR4(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type R4 """
    if not checkArray(a):
        return False
    if a.dtype != __R4:
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayR8(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type R8 """
    if not checkArray(a):
        return False
    if a.dtype != __R8:
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayReal(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type R4 or R8"""
    if not checkArray(a):
        return False
    if a.dtype not in [__R4, __R8]:
        return False
    return True


# -----------------------------------------------------------------------------
def checkArrayInteger(a, dienow=False):
    """same as :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>` for an array of type I4,I8"""
    if not checkArray(a):
        return False
    if a.dtype not in [__I4, __I8]:
        return False
    return True

# -----------------------------------------------------------------------------
def checkSingleValue(a):
    """checks if a value is a single value, that is a single integer,
    a single float or a single string"""
    if checkArrayReal(a) or checkArrayInteger(a):
        if a.shape == (1,):
            return True
        return False
    elif checkArrayChar(a):
        if len(a.shape) == 1:
            return True
        return False
    return False

# -----------------------------------------------------------------------------
def checkNodeCompliant(node, parent=None, dienow=False):
    """
    Performs all possible checks on a node. Can raise any of the exceptions
    related to node checks (:py:func:`checkNodeName`, :py:func:`checkNodeType`,
    :py:func:`checkArray`...)::

       if not checkNodeCompliant(node):
           raise Exception('Bad Node')

    :arg CGNS/Python node: the CGNS/Python node to check
    :arg CGNS/Python parent: CGNS/Python parent node to check (if not None)
    :return: True if all controls are ok
    :Remarks:
      - Calls :py:func:`checkNode`,:py:func:`checkNodeName`,
        :py:func:`checkArray`, :py:func:`checkNodeType`

    """
    r = checkNode(node, dienow=dienow) \
        and checkNodeName(node, dienow=dienow) \
        and checkArray(node[1], dienow=dienow) \
        and checkNodeType(node, dienow=dienow)
    return r


# -----------------------------------------------------------------------------
def concatenateForArrayChar(nlist):
    nl = []
    for n in nlist:
        if isinstance(n, str):
            nl += [setStringAsArray(("%-32s" % n)[:32])]
        else:
            checkArrayChar(n)
            nl += [setStringAsArray(("%-32s" % n.tostring())[:32])]
    r = numpy.array(numpy.array(nl, order='F').T, order='F')
    return r


# -----------------------------------------------------------------------------
def concatenateForArrayChar2D(nlist):
    """Creates a numpy.ndarray of chars from a list of python strings::

       udims=['Kilogram','Meter','Second','Kelvin','Gradian']
       a=concatenateForArrayChar2D(udims)

    The result is a numpy.ndarray of type 'S' with a shape of (32,N) for
    N strings. In the example above the value of a[:,1] is 'Meter' with
    an added padding of 27 'spaces'.
    The order of the values in the second axis is kept unchanged.
    """
    return concatenateForArrayChar(nlist)


# -----------------------------------------------------------------------------
def concatenateForArrayChar3D(nlist):
    """Creates a numpy.ndarray of chars from a list of list of python strings"""
    rr = []
    for p in nlist:
        nl = []
        for n in p:
            if isinstance(n, str) :
                nl += [setStringAsArray(("%-32s" % n)[:32])]
            else:
                checkArrayChar(n)
                nl += [setStringAsArray(("%-32s" % n.tostring())[:32])]
        rr.append(numpy.array(numpy.array(nl, order='F'), order='F'))
    r = numpy.array(rr, order='F').T
    return r


# -----------------------------------------------------------------------------
# old MLL returns - should not be used anymore
def getValueType(v):
    """
    Returns the node's value type as CGNS/MLL type.
    The return value is a string in:
    Character, RealSingle, RealDouble, Integer, LongInteger
    """
    if not isinstance(v, numpy.ndarray):
        return None
    if v.dtype.kind in ['S', 'a']:
        return CK.Character_s
    if v.dtype.name in ['float32']:
        return CK.RealSingle_s
    if v.dtype.name in ['float64']:
        return CK.RealDouble_s
    if v.dtype.name in ['int32']:
        return CK.Integer_s
    if v.dtype.name in ['int64']:
        return CK.LongInteger_s
    return None


# -----------------------------------------------------------------------------
def setValue(node, value):
    """
    Sets an arbitrary value in the node, replacing the existing one.
    The value is refered-to, no copy, data type is deduced from numpy array
    data type."""
    if node is None:
        return None
    if not isinstance(value, numpy.ndarray):
        node[1] = None
    elif (value.dtype.kind in ['S', 'a'] or
        value.dtype.name in ['float32', 'float64', 'int32', 'int64']):
        node[1] = value
    else:
        return None
    return node


# -----------------------------------------------------------------------------
def setStringByPath(tree, path, s):
    """
    Creates a 1D numpy.ndarray from one string, set to node by path::

      p='/{Base#1}/BaseIterativeData/DataClass'
      setStringByPath(tree,p,'UserDefined')

    Remark: target node should already exist
    """
    node = getNodeByPath(tree, path)
    v = setStringAsArray(s)
    if v is not None:
        setValue(node, v)
    return node

# -----------------------------------------------------------------------------
def setStringAsArray(a):
    """Creates a numpy.ndarray from a string::

       setStringAsArray('UserDefined')

    """
    if isinstance(a, numpy.ndarray):
        if (a.shape != ()) and (a.dtype.kind == 'S'):
            return a
        else:
            return numpy.array(tuple(a), dtype='|S', order='F')
    if isinstance(a, bytes):
        lst_bytes = [bytes([x]) for x in a]
        if not PY3:
            lst_bytes = [x.decode('ascii') for x in lst_bytes]
        return numpy.array(lst_bytes, dtype='|S', order='F')
    if isinstance(a, str) :
        lst_bytes = [x.encode('ascii') for x in a]
        if not PY3:
            lst_bytes = [x.decode('ascii') for x in lst_bytes]
        return numpy.array(lst_bytes, dtype='|S', order='F')
    return None

# -----------------------------------------------------------------------------
def setIntegerByPath(tree, path, *i):
    """Creates a 1D numpy.ndarray from one or more integers,
       set to node by path. Same as :py:func:`setLongByPath` but with a
       numpy.int32 data type.

       Remark: target node should already exist
    """
    if isinstance(i, type(None)) or (len(i) == 0):
        raise CE.cgnsNameError(112)
    node = getNodeByPath(tree, path)
    setValue(node, numpy.array(i, dtype=numpy.int32))
    return node

# -----------------------------------------------------------------------------
def setIntegerAsArray(*i):
    """Creates a 1D numpy.ndarray from one or more integers.
       Same as :py:func:`setLongAsArray` but with a
       numpy.int32 data type.

    """
    if isinstance(i, type(None)) or (len(i) == 0):
        raise CE.cgnsNameError(112)
    return numpy.array(i, dtype=numpy.int32)

# -----------------------------------------------------------------------------
def setLongByPath(tree, path, *l):
    """Creates a 1D numpy.ndarray from one or more longs,
       set to node by path::

           p='/{Base#1}/BaseIterativeData/NumberOfZones'
           setFloatByPath(tree, p, 2)

           p='/{Base#1}/BaseIterativeData/IterationValues'
           setFloatByPath(tree, p, 1, 2, 3, 4, 5)
           setFloatByPath(tree, p, tuple(range(10,1010,10)))

       The set value has numpy.int64 data type

       Remark: target node should already exist
    """
    if isinstance(l, type(None)) or (len(l) == 0):
        raise CE.cgnsNameError(112)
    node = getNodeByPath(tree, path)
    setValue(node, numpy.array(l, dtype=numpy.int64))
    return node

# -----------------------------------------------------------------------------
def setLongAsArray(*l):
    """Creates a 1D numpy.ndarray from one or more longs::

         setLongAsArray(2)
         setLongAsArray(1, 2, 3, 4, 5)
         setLongAsArray(tuple(range(10,1010,10)))

       The set value has numpy.int64 data type
    """
    if isinstance(l, type(None)) or (len(l) == 0):
        raise CE.cgnsNameError(112)
    return numpy.array(l, dtype=numpy.int64)

# -----------------------------------------------------------------------------
def setFloatByPath(tree, path, *f):
    """Creates a 1D numpy.ndarray from one or more floats,
    set to node by path::

      p='/{Base#1}/{Zone-A}/ReferenceState/Coef_PressureDynamic'
      setFloatByPath(tree, p, 2837.153)

      p='/{Base#1}/{Zone-A}/ReferenceState/Coef_Local'
      setFloatByPath(tree, p, 2.1, 2.2, 2.3, 2.4)

    The set value has numpy.float32 data type

    Remark: target node should already exist
    """
    if isinstance(f, type(None)) or (len(f) == 0):
        raise CE.cgnsNameError(112)
    node = getNodeByPath(tree, path)
    setValue(node, numpy.array(f, dtype=numpy.float32))
    return node

# -----------------------------------------------------------------------------
def setFloatAsArray(*f):
    """Creates a 1D numpy.ndarray from one or more floats::

         setFloatAsArray(2837.153)
         setFloatAsArray(2.1, 2.2, 2.3, 2.4)
         setFloatAsArray(tuple(range(10,1010,10)))

       The returned array has numpy.float32 data type
    """
    if isinstance(f, type(None)) or (len(f) == 0):
        raise CE.cgnsNameError(112)
    return numpy.array(f, dtype=numpy.float32)


# -----------------------------------------------------------------------------
def setDoubleByPath(tree, path, *d):
    """Creates a 1D numpy.ndarray from one or more doubles,
      set to node by path. Same as :py:func:`setFloatByPath` but with a
      numpy.float64 data type.

      Remark: target node should already exist
    """
    if isinstance(d, type(None)) or (len(d) == 0):
        raise CE.cgnsNameError(112)
    node = getNodeByPath(tree, path)
    setValue(node, numpy.array(d, dtype=numpy.float64))
    return node


# -----------------------------------------------------------------------------
def setDoubleAsArray(*d):
    """Creates a 1D numpy.ndarray from one or more doubles.
    Same as :py:func:`setFloatAsArray` but with a
    numpy.float64 data type.

    """
    if isinstance(d, type(None)) or (len(d) == 0):
        raise CE.cgnsNameError(112)
    return numpy.array(d, dtype=numpy.float64)


# -----------------------------------------------------------------------------
def getValueAsString(node):
    """Returns node value as a Python string"""
    return node[1].tostring().decode("ascii", "strict")


# -----------------------------------------------------------------------------
def getValueAsStringEval(value):
    """Tries to build the best value one can deduce from given string, the
    returned value is a numpy array or None (if everything else fail).
    Evaluation is performed on string with the following order:

     - numpy array of single int
     - numpy array of single float
     - numpy array of ints
     - numpy array of floats
     - numpy array of chars
    """
    if eval(value) is None:
        return None
    try:
        v = numpy.array(value, dtype=numpy.int64)
        return v
    except ValueError:
        pass

    try:
        v = numpy.array(value, dtype=numpy.float64)
        return v
    except ValueError:
        pass

    try:
        v = numpy.array(eval(value), dtype=numpy.int64)
        return v
    except ValueError:
        pass

    try:
        v = numpy.array(eval(value), dtype=numpy.float64)
        return v
    except ValueError:
        pass
    try:
        v = setStringAsArray(value)
        return v
    except ValueError:
        pass

    return None


# -----------------------------------------------------------------------------
def getValue(node):
    """Returns node value, could be `None` or a `numpy.ndarray`."""
    v = node[1]
    if not isinstance(v, numpy.ndarray):
        return None
    if (v.dtype.kind in ['S', 'a'] or
        v.dtype.name in ['float32', 'float64', 'int32', 'int64']):
        return v
    else:
        return None


# --------------------------------------------------
def hasFortranFlag(node):
    """Returns node value fortran flag."""
    if node[1] is None:
        return True
    if node[1] == []:
        return True
    if isinstance(node[1], str):
        return True  # link
    if not node[1].shape:
        return True
    if len(node[1].shape) == 1:
        return True
    return numpy.isfortran(node[1])


# --------------------------------------------------
def getValueShape(node):
    """
    Returns the value data shape for a CGNS/Python node for **display purpose**.
    If the shape cannot be determined a `-` is returned::

       print 'node data shape is %s'%getValueShape(node)

    :arg CGNS/Python node: the target node
    :return: A string with the shape

    """
    return getNodeShape(node)


def getNodeShape(node):
    if node[1] is None:
        r = "-"
    elif node[1] == []:
        r = "-"
    elif node[3] == '':
        r = "-"
    elif node[1].shape in ['', (0,), ()]:
        r = "[0]"
    else:
        r = str(list(node[1].shape))
    return r


def getShape(node):
    if not isinstance(node[1], numpy.ndarray):
        r = (0,)  # None, []
    elif node[3] == '':
        r = (0,)
    elif node[1].shape in ['', (0,), ()]:
        r = (0,)
    else:
        r = node[1].shape
    return r


# --------------------------------------------------
def getNodeAllowedChildrenNames(node):
    return getAuthNames(node)


# --------------------------------------------------
def getAuthNames(node):
    """
    Returns the authorized names for a CGNS/Python node.
    If the names cannot be determined an empty list is returned::

       node=['gasmodel',None,[],'GasModel_t']
       if (node[0] not in getAuthNames(node)):
           print 'not SIDS compliant name'

    :arg CGNS/Python node: the target node
    :return: A list of authorized names (empty list if not found)
    """
    if not node[3]:
        r = []  # ''
    elif CT.types[node[3]].names in ['', None, CT.UD]:
        r = []
    else:
        r = CT.types[node[3]].names
    return r


# --------------------------------------------------
def getAuthDataTypes(node):
    """
    Returns the authorized types for a CGNS/Python node.
    If the types cannot be determined a None is returned::

       if (getValueType(node) not in getAuthDataTypes(node)):
           print 'Type of node value is not SIDS comliant'

    :arg CGNS/Python node: the target node
    :return: A list of authorized data types (empty if not found)

    """
    if node[1] is None:
        r = []
    elif node[1] == []:
        r = []
    elif not node[3]:
        r = []  # ''
    elif CT.types[node[3]].datatype in ['', None, [CK.LK]]:
        r = []
    else:
        r = CT.types[node[3]].datatype
    return r


# --------------------------------------------------
def getAuthParentTypes(node):
    """
    Returns the authorized parent types for a CGNS/Python node.
    If the parent types cannot be determined a None is returned::

        np=getParentFromNode(T,node)
        if (np[3] not in getAuthParentTypes(node)):
            p=getPathByNode(T,node)
            print '[%s] cannot have parent of type [%s]'%(p,np[3])

    :arg CGNS/Python node: the target node
    :return: A list of authorized parent types (empty list is none)

    """
    if node[1] is None:
        r = []
    elif node[1] == []:
        r = []
    elif not node[3]:
        r = []
    elif CT.types[node[3]].parents in ['', None]:
        r = []
    else:
        r = CT.types[node[3]].parents
    return r


# --------------------------------------------------
def getAuthShapes(node):
    """Returns the authorized shapes for a CGNS/Python node.
    If the shapes cannot be determined a None is returned::

       if getShape(node) not in getAuthShapes(node):
           p=getPathByNode(T,node)
           print '[%s] cannot have shape [%s]'%(p,getShape(node))

    :arg CGNS/Python node: the target node
    :return: A list of authorized shapes

    """
    if node[1] is None:
        r = []
    elif node[1] == []:
        r = []
    elif not node[3]:
        r = []
    elif CT.types[node[3]].shape in ['']:
        r = []
    else:
        r = CT.types[node[3]].shape
    return r


# --------------------------------------------------
def getAuthChildren(node):
    """
    Returns the authorized children for a CGNS/Python node.
    If the children types cannot be determined a None is returned::

       if hasChildNodeOfType(node) not in getAuthChildren(node):
           p = getPathByNode(T,node)
           print '[%s] cannot have [%s] of type [%s] as child'%(p,node[0],node[3])

    :arg node node: target node
    :return: list of str, authorized CGNS/SIDS types for children

    """
    if not node[3]:
        r = []  # None, [], ''
    elif CT.types[node[3]].children in ['', None, []]:
        r = []
    else:
        r = CT.types[node[3]].children
    return r


# --------------------------------------------------
def getValueDataType(node):
    """
    Returns the value data type for a CGNS/Python node for **display purpose**::

       print 'node data type is %s'%getValueDataType(node)

    :arg CGNS/Python node: function target
    :return: A data type as string in ``[`C1`,`I4`,`I8`,`R4`,`R8`,`??`]``
    :Remarks:
      - ``??`` returned if datatype is not one of ``[`C1`,`I4`,`I8`,`R4`,`R8`]``
      - Then ``None`` value returns ``??``
      - There is no ``LK`` link type with the CGNS/Python mapping

    """
    return getNodeType(node)


# --------------------------------------------------
def getNodeType(node):
    data = node[1]
    if node[0] == 'CGNSLibraryVersion_t':
        return CK.R4  # ONLY ONE R4 IN ALL SIDS !
    if isinstance(data, numpy.ndarray):
        if data.dtype.kind in ['S', 'a']:
            return CK.C1
        if data.dtype.name in ['float32', 'Float32']:
            return CK.R4
        if data.dtype.name in ['float64', 'Float64']:
            return CK.R8
        if data.dtype.name in ['int32', 'Int32']:
            return CK.I4
        if data.dtype.name in ['int64', 'Int64']:
            return CK.I8
    if isinstance(data, list) and (len(data)):  # oups !
        if isinstance(data[0], str):
            return CK.C1
        if isinstance(data[0], int):
            return CK.I4
        if isinstance(data[0], float):
            return CK.R8
    if (data is None) or (data == []):
        return CK.MT
    return '??'


# --------------------------------------------------
def hasFirstPathItem(path, item=CK.CGNSTree_s):
    """
    True if the arg string is the item of the path::

      p='/{Base#1}/{Zone-A}/ZoneGridConnectivity'
      print hasFirstPathItem(p,'{Base#1}')
      # True

    :arg str path: path to process
    :arg str item: item to check against
    :return bool: True if first item matches arg
    :Remarks:
      - There is no call to :py:func:`getPathNormalize`
      - Default value for item is `CGNSTree`

    """
    if (len(path) > 0) and (path[0] == '/'):
        path = path[1:]
    p = path.split('/')
    if (len(p) > 1) and (item == p[0]):
        return True
    return False


# --------------------------------------------------
def removeFirstPathItem(path):
    """
    Return the path without the first item::

      print removeFirstPathItem('/{Base#1}/{Zone-A}/ZoneGridConnectivity')
      # '/{Zone-A}/ZoneGridConnectivity'

    :arg str path: path to process
    :return: path without first token
    :Remarks:
      - There is no call to :py:func:`getPathNormalize`.

    """
    p = path.split('/')
    if (p[0] == '') and (len(p) > 2):
        return str.join('/', [''] + p[2:])
    elif len(p) > 1:
        return str.join('/', p[1:])
    else:
        return '/'


# --------------------------------------------------
def getNodeByPath(tree, path):
    """
    Returns the CGNS/Python node with the argument path::

      zbc=getNodeByPath(tree,'/Base/Zone001/ZoneBC')
      nchildren=len(childrenNames(zbc))

    The path is compared as a string, you should provide the exact path
    if you have a sub-tree or a tree with its `CGNSTree` fake node. The
    following lines are not equivalent (sic!)::

      zbc=getNodeByPath(tree,'/Base/Zone001/ZoneBC')
      zbc=getNodeByPath(tree,'/CGNSTree/Base/Zone001/ZoneBC')

    You can change the relative root by giving any sub-node of the complete tree.
    For example, to get a specific BC node in a zone, you first look for the
    ``ZoneBC`` of the zone and you use the returned node a the new root::

      zbc=getNodeByPath(tree,'/Base/Zone001/ZoneBC')
      nbc=getNodeByPath(zbc,'./wall01')

    :arg node tree: the target tree to parse
    :arg str path: absolute or relative path
    :return:
      - The CGNS/Python `node` matching the path
      - Returns `None` if the path is not found
    :Remarks:
      - No wildcards allowed (see :py:func:`getPathsByNameFilter`
        and :py:func:`getPathsByNameFilter` )
      - there is no concept of absolute or relative path, the path is always the
        concatenation of children node names (and then recurse)
    """
    path = getPathNormalize(path)
    if path in ['', '/', '.']:
        return tree
    if not checkPath(path):
        return None
    lpath = path.split('/')
    if lpath[0] == '':
        lpath = lpath[1:]
    if tree[3] == CK.CGNSTree_ts:
        _tree = tree
        if lpath[0] == CK.CGNSTree_s:
            if len(lpath) == 1:
                return _tree
            lpath = lpath[1:]
    elif lpath[0] == tree[0]:
        _tree = [CK.CGNSTree_s, None, [tree], CK.CGNSTree_ts]
    else:
        _tree = tree
    n = getNodeFromPath(lpath, _tree)
    return n


# --------------------------------------------------
def getValueByPath(tree, path):
    """
    Returns the value of a CGNS/Python node with the argument path::

      import CGNS.PAT.cgnskeywords as CK

      v=getNodeByPath(T,'/Base/Zone001/ZoneType')
      if (v == CK.Structured_s): print 'Structured Zone Found'

    :arg CGNS/Python tree: target tree to parse
    :arg str path: absolute or relative path
    :return:
      - CGNS/Python node value matching the path
      - Returns None if the path is not found
    :Remark:
      - No wildcards allowed (see :py:func:`getPathsByNameFilter`
        and :py:func:`getPathsByNameFilter` )
    """
    n = getNodeByPath(tree, path)
    if n is None:
        return None
    return n[1]


# --------------------------------------------------
def getChildrenByPath(tree, path):
    """
    Returns the children list of a CGNS/Python node with the argument path::

      import CGNS.PAT.cgnskeywords as CK

      for bc in getChildrenByPath(T,'/Base/Zone01/ZoneBC'):
          if (bc[3] == CK.BC_ts):
              print 'BC found:', bc[0]

    :arg CGNS/Python tree: target tree to parse
    :arg str path: absolute or relative path
    :return:
      - The CGNS/Python node children list of node matching the path
      - Returns None if the path is not found
    :Remark:
      - No wildcards allowed (see :py:func:`getPathsByNameFilter`
        and :py:func:`getPathsByNameFilter` )
    """
    n = getNodeByPath(tree, path)
    if n is None:
        return None
    return n[2]


# --------------------------------------------------
def getNextChildSortByType(node, parent=None, criteria=None):
    """
    **Iterator** returns the children list of the argument CGNS/Python
    sorted using the CGNS type then the name. The `sortlist` gives
    an alternate sort list/dictionnary::

      for child in getNextChildSortByType(node):
          print 'Next child:', child[0]

      zonesort=[CGK.Elements_ts, CGK.Family_ts, CGK.ZoneType_ts]
      for child in getNextChildSortByType(node,criteria=mysort):
          print 'Next child:', child[0]

      mysort={CGK.Zone_t: zonesort}
      for child in getNextChildSortByType(node,parent,mysort):
          print 'Next child:', child[0]

    :arg CGNS/Python node: the target
    :arg CGNS/Python parent: the parent
    :arg list criteria: a list or a dictionnary used as the sort criteria
    :return:
      - This is an iterator, it returns a CGNS/Python node
    :remarks:
      - The function is an **iterator**
      - If criteria is a list of type, the sort order for the type is the
        list order. If it is a dictionnary, its keys are the parent types
        and the values are list of types.
      - Default value for criteria is CGNS.PAT.cgnskeywords.cgnstypes
    """

    def sortbytypesasincriteria(a, b):
        if (a[0] in a[2]) and (b[0] in b[2]):
            if a[2].index(a[0]) > b[2].index(b[0]):
                return 1
            if a[2].index(a[0]) < b[2].index(b[0]):
                return -1
        if a[1] > b[1]:
            return 1
        if a[1] < b[1]:
            return -1
        return 0

    if criteria is None:
        criteria = CK.cgnstypes
    __criteria = []
    if isinstance(criteria, list):
        __criteria = criteria
    if (isinstance(criteria, dict) and
            (parent is not None) and
            (parent[3] in criteria)):
        __criteria = criteria[parent[3]]
    r = []
    for i, c in enumerate(node[2]):
        r += [(c[3], c[0], __criteria, i)]
    r.sort(key=cmp_to_key(sortbytypesasincriteria))
    for i in r:
        yield node[2][i[3]]


# --------------------------------------------------
def getTypeByPath(tree, path):
    """
    Returns the CGNS type of a CGNS/Python node with the argument path::

      import CGNS.PAT.cgnskeywords as CK

      if (getTypeByPath(T,'/Base/Zone01/ZoneBC/')):
          if (bc[3] == CK.BC_ts):
              print 'BC found:', bc[0]

    :arg CGNS/Python tree: target tree to parse
    :arg str path: absolute or relative path
    :return:
      - the CGNS/SIDS type (str)
      - None if the path is not found
    :remark:
      - No wildcards allowed (see :py:func:`getPathsByTypeFilter`
        and :py:func:`getPathsByNameFilter` )
    """
    n = getNodeByPath(tree, path)
    if n is not None:
        return n[3]
    return None


# --------------------------------------------------
def getPathByNameFilter(tree, filter):
    return getPathsByNameFilter(tree, filter)


# --------------------------------------------------
def getPathsByTokenFilter(tree, filter):
    """
    Returns a list of paths from T matching the filter. The filter is a
    `regular expression <http://docs.python.org/library/re.html>`_
    used to match at least one of the token of the path::

     import CGNS.PAT.cgnskeywords as CK

     for path in getPathsByTokenFilter(T,'Family.*'):
         print 'Family ',path,' is ',path[2]

    :arg CGNS/Python tree: target tree to parse
    :arg str filter: a regular expression for the token to match to
    :return:
      - A list of paths (strings) matching the path pattern
      - Returns empty list if no match
    :Remarks:
      - You cannot use the regex to match for a path
      - Always skips `CGNSTree_t`
    """
    lpth = getAllPaths(tree)
    reg = re.compile(filter)
    rpth = []
    for p in lpth:
        pl = getPathToList(p, True)
        for tk in pl:
            if reg.match(tk) is not None:
                rpth.append(p)
    return rpth


# --------------------------------------------------
def getPathsByNameFilter(tree, filter):
    """
    Returns a list of paths from T matching the filter. The filter is a
    `regular expression <http://docs.python.org/library/re.html>`_
    used to match the path of **node names**::

     import CGNS.PAT.cgnskeywords as CK

     for path in getPathsByNameFilter(T,'/Base[0-1]/domain\..*/.*/.*/FamilyName'):
        print 'FamilyName ',path,' is ',path[2]

    :arg CGNS/Python tree: target tree to parse
    :arg str filter: a regular expression for the complete path to match to
    :return:
      - A list of paths (strings) matching the path pattern
      - Returns empty list if no match
    :Remarks:
      - The '/' is the separator for the path tokens, so you cannot use it
        in the regular expression for any other purpose. Then you cannot match
        any number of tokens in the filter, you must have the exact count of
        '/' chars.
      - Always skips `CGNSTree_t`
    """
    recmplist = []
    restrlist = filter.split('/')[1:]
    for restr in restrlist:
        recmplist.append(re.compile(restr))
    lpth = getAllPaths(tree)
    rpth = []
    for p in lpth:
        pl = getPathToList(p, True)
        if len(pl) != len(recmplist):
            continue
        for n, pattern in enumerate(recmplist):
            if pattern.match(pl[n]) is None:
                break
        else:
            rpth.append(p)
    return rpth


# --------------------------------------------------
def getPathAsTemplate(sidslist, namelist, required=None, enclosed=False):
    """
    Return a list of CGNS/SIDS type or name (strings) by replacing in a path
    all the user defined names by their CGNS/SIDS type. If the name is a
    non-ambiguous SIDS name, it is kept as name (for example GridLocation and
    GridLocation_t, ZoneBC_t and ZoneBC...).
    The required list of boolean forces the name to be kept if True.
    For example:

      sidslist=[CGNSBase_t, Zone_t, ZoneBC_t, BC_t, GridLocation_t]
      namelist=[Base#1, Zone#1, ZoneBC, BC#1, GridLocation]

      returns=[CGNSBase_t, Zone_t, ZoneBC, BC_t, GridLocation]

      # add a required list
      required=[False, True, False, False, False]
      returns=[CGNSBase_t, Zone#1, ZoneBC, BC_t, GridLocation]

    This function is used for pattern storage/check.

    Return [] in failure (should be an except).
    """
    if len(sidslist) != len(namelist):
        return []
    if not required:
        required = [False] * len(namelist)

    r = []
    for idx, name in enumerate(namelist):
        if required[idx]:
            r += [name]
        else:
            sids = sidslist[idx]
            e = CT.types[sids]
            if enclosed:
                sids = '{%s}' % sids
            if CT.UD not in e.names:
                if name in e.names:
                    r += [name]
                else:
                    r += [sids]
            else:
                r += [sids]
    return r


# --------------------------------------------------
def getPathByTypeFilter(tree, filter):
    return getPathsByTypeFilter(tree, filter)


# --------------------------------------------------
def getPathsByTypeFilter(tree, filter):
    """
    Returns a list of paths from T matching the filter. The filter is a
    `regular expression <http://docs.python.org/library/re.html>`_
    used to match the path of **node types**::

      # gets GridConnectivity_t and GridConnectivity1to1_t
      allconnectivities=getPathsByTypeFilter(T,'/.*/.*/.*/GridConnectivity.*')

    :arg node tree: the target tree to parse
    :arg str filter: a regular expression for the complete path to match to

    :Return:
      - A list of paths (str) matching the types-path pattern
      - Returns empty list if no match

    :Remarks:
      - The '/' is the separator for the path tokens, so you cannot use it
       in the regular expression for any other purpose
      - Always skips `CGNSTree_t`
    """
    recmplist = []
    restrlist = filter.split('/')[1:]
    for restr in restrlist:
        recmplist.append(re.compile(restr))
    lpth = getAllPaths(tree)
    rpth = []
    for p in lpth:
        pl = getPathAsTypes(tree, p)[1:]
        if len(pl) != len(recmplist):
            continue
        for n, pattern in enumerate(recmplist):
            if isinstance(pl[n], str) and (pattern.match(pl[n]) is None):
                break
        else:
            rpth.append(p)
    return rpth


# --------------------------------------------------
def nodeByPath(path, tree):
    if not checkPath(path):
        return None
    if path[0] == '/':
        path = path[1:]
    if tree[3] == CK.CGNSTree_ts:
        n = getNodeFromPath(path.split('/'), tree)
    else:
        n = getNodeFromPath(path.split('/'), [None, None, [tree], None])
    return n


# --------------------------------------------------
def removeChildByName(parent, name):
    """Remove the child from the parent node.

    :arg CGNS/Python node parent: node where to find the child name
    :arg str name: name of the child to delete (with all its sub-tree)
    :return: None
    :Remarks:
      - node name to delete is a direct child of the parent node
      - See also :py:func:`nodeDelete`
    """
    if not checkNode(parent):
        return None
    for n, node in enumerate(parent[2]):
        if node[0] == name:
            del parent[2][n]
            return None
    return None


# --------------------------------------------------
def removeNodeFromPath(path, node):
    target = getNodeFromPath(path, node)
    if len(path) > 1:
        father = getNodeFromPath(path[:-1], node)
        father[2].remove(target)
    else:
        # Root node child
        for c in node[2]:
            if c[0] == path[0]:
                node[2].remove(target)


# --------------------------------------------------
def getNodeFromPath(path, node):
    # Beware: this parse starts with children, not current node...
    for c in node[2]:
        if c[0] == path[0]:
            if len(path) == 1:
                return c
            return getNodeFromPath(path[1:], c)
    return None


# --------------------------------------------------
def getParentFromNode(tree, node):
    """
    Returns the parent node of a node. If the node is root node, itself is
    returned::

      parent=getParentFromNode(T,node)

    :arg CGNS/Python tree: target tree to parse
    :arg CGNS/Python node: child node
    :return:
      - the parent node
      - arg ``node`` itself if node is root
    """
    pn = getPathFromNode(tree, node)
    pp = getPathAncestor(pn)
    np = getNodeByPath(tree, pp)
    return np


# --------------------------------------------------
def getAncestorByType(tree, node, ptype):
    """
    Returns the parent node of a node which has the CGNS/SIDS type. If the node is root node, itself is
    returned::

      >>> import CGNS.PAT.cgnskeywords as CGK
      >>> base = getAncestorByType(tree, node, CGK.CGNSBase_ts)

    :arg CGNS/Python tree: target tree to parse
    :arg CGNS/Python node: child node
    :arg CGNS/SIDS ptype: required type of the parent
    :return:
      - the parent node with CGNS/SIDS type
      - arg ``node`` itself if node is root
      - None if CGNS/SIDS type not in parents node
    """
    pn = getPathFromNode(tree, node)
    pln = getPathToList(pn)
    ptn = getPathAsTypes(tree, pn, legacy=False)
    try:
        i = ptn.index(ptype)
    except ValueError:
        return None
    if not checkRootNode(tree):
        i += 1
    pp = "/".join(pln[:i + 1])
    np = getNodeByPath(tree, pp)
    return np


# --------------------------------------------------
def getPathFromRoot(tree, node):
    """
    Same as :py:func:`getPathFromNode` but takes into account the root
    node name::

      n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
      r=CGU.nodeCreate('State',None,[],CGK.ReferenceState_ts,parent=n)
      d=CGU.nodeCreate('Data',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)

      CGU.getPathFromRoot(n,d)
      # '/Base/State/Data'
      CGU.getPathFromRoot(r,d)
      # '/Base/Data'

    :arg CGNS/Python tree: target tree to parse
    :arg CGNS/Python node: target node
    :return:
      - the path of the node (str)
    """
    return getPathFromNode(tree, node, '/' + tree[0])


# --------------------------------------------------
def getPathFromNode(tree, node, path=''):
    """
    Returns the path from a node in a tree. The argument tree is parsed and
    a path is built-up until the node is found. Then the **start node** name
    is not taken into account. For example::

      n=CGU.nodeCreate('Base',numpy.array([3,3]),[],CGK.CGNSBase_ts)
      r=CGU.nodeCreate('State',None,[],CGK.ReferenceState_ts,parent=n)
      d=CGU.nodeCreate('Data',numpy.array([3.14]),[],CGK.DataArray_ts,parent=r)

      CGU.getPathFromNode(n,d)
      # '/State/Data'
      CGU.getPathFromNode(r,d)
      # '/Data'

    In the case you want to add the name of the root node (**start node**) in
    the path, you should use the ``path`` argument
    (see also :py:func:`getPathFromRoot`)::

      CGU.getPathFromNode(n,d,'/'+n[0])
      # '/Base/ReferenceState/Data'

    The functions behaves like this for historical reasons, the root of
    a CGNS/Python tree is ``CGNSTree`` but is not CGNS/SIDS compliant. So the
    path of a ``CGNSBase_t``, starting from the root node, is `/Base`
    instead of the logically expected `/CGNSTree/CGNSBase`.

    The node object is compared to the tree nodes, if you have multiple
    references to the same node, the first found is used for the path::

     # T is a compliant CGNS/Python tree
     path=getPathFromNode(T,node)
     getNodeByPath(T,getPathAncestor(path))

    :arg CGNS/Python tree: target tree to parse
    :arg CGNS/Python node: target node to find
    :arg string path: name of the root node to add if desired
    :return:
      - path as string
      - None if not found
    :remark:
      - see also :py:func:`getPathFromRoot`
    """
    if id(node) == id(tree):
        return path
    for c in tree[2]:
        p = getPathFromNode(c, node, path + '/' + c[0])
        if p:
            return p
    return None


# --------------------------------------------------
def zipTypeOrNameList(tlist, nlist):
    """
    Mixes two lists for pattern search: the CGNS/SIDS type list and the
    name list.

    For example with two lists::

     >>> for q in zipTypeOrNameList([1,2,3,4],['a','b','c','d']): print(q)
     [1, 2, 3, 4]
     [1, 2, 3, 'd']
     [1, 2, 'c', 4]
     [1, 2, 'c', 'd']
     [1, 'b', 3, 4]
     [1, 'b', 3, 'd']
     [1, 'b', 'c', 4]
     [1, 'b', 'c', 'd']
     ['a', 2, 3, 4]
     ['a', 2, 3, 'd']
     ['a', 2, 'c', 4]
     ['a', 2, 'c', 'd']
     ['a', 'b', 3, 4]
     ['a', 'b', 3, 'd']
     ['a', 'b', 'c', 4]
     ['a', 'b', 'c', 'd']

    """
    w = []
    r = []
    z = len(tlist)
    for b in range(2 ** z):
        w.append([int(x) for x in '{0:0{width}{base}}'.format(b, base='b', width=z)])
    for b in w:
        h = []
        for u, item in enumerate(b):
            if item:
                h.append(nlist[u])
            else:
                h.append(tlist[u])
        r.append(h)
    return r


# --------------------------------------------------
def getAllNodesByTypeOrNameList(tree, typeornamelist):
    return getPathsByTypeOrNameList(tree, typeornamelist)


# --------------------------------------------------
def getPathsByTypeOrNameList(tree, typeornamelist):
    """
    Returns a list of paths from the argument tree with nodes matching
    the list of types or names. The list you give is the list you would
    have if you pick the node type or the node name during the parse::

      tnlist=['CGNSTree_t','Base#001','Zone_t']

      for path in getPathsByTypeOrNameList(T,tnlist):
          node=getNodeByPath(T,path)
          # do something with node

    Would return all the zones of the named base.
    See also :py:func:`getPathsByTypeSet`
    See also :py:func:`getPathsByTypeList`

    :arg CGNS/Python tree: the start node of the CGNS tree to parse
    :arg list typeornamelist: the (ordered) list of CGNS/SIDS types
    :return: a list of strings, each string is the path to a matching node
    :remarks:
      - only plain strings allowed, no regular expression
      - the first comparison is performed on name, then on type. If you have
        a node name that matches a type, the node is included in the result.
    """
    if (tree[0] != typeornamelist[0]) and (tree[3] != typeornamelist[0]):
        return []
    if tree[3] == CK.CGNSTree_ts:
        start = ""
    else:
        start = "%s" % tree[0]
    n = getAllNodesFromTypeOrNameList(typeornamelist[1:], tree[2], start, [])
    return n


# --------------------------------------------------
def getAllParentTypePathsAux(ntype, path, plist):
    tlist = CT.types[ntype].parents
    for pt in tlist:
        plist.append(path + '/' + pt)
        getAllParentTypePathsAux(pt, path + '/' + pt, plist)


# --------------------------------------------------
def getAllParentTypePaths(nodetype):
    return getAuthParentTypePaths(nodetype)


# --------------------------------------------------
def getAuthParentTypePaths(nodetype):
    """
    Return all type paths allowed by CGNS/SIDS for the node type.
    For example, to check all possible places where you can set a
    `FamilyName_t` you call `getAllParentTypePaths`, it returns you a list
    of all types paths allowed by CGNS/SIDS.

    :arg str nodetype: a CGNS/SIDS type as string
    :return: a list of list of strings, each string is a CGNS/SIDS type path
    :Remarks:
      - You can loop on this list of list to feed input arguments for a call to
        :py:func:`getPathsByTypeList` for example.
      - See also :py:func:`getPathAsTypes`

    """
    r = []
    p = ''
    l = []
    getAllParentTypePathsAux(nodetype, p, r)
    for tp in r:
        s = tp.split('/')
        if s[-1] == CK.CGNSTree_ts:
            s.reverse()
            s[-1] = nodetype
            l.append(s)
    return l


# --------------------------------------------------
def getAllNodesFromTypeOrNameList(tnlist, node, path, result):
    if not tnlist:
        return result
    for c in node:
        if (c[0] == tnlist[0]) or (c[3] == tnlist[0]):
            if len(tnlist) == 1:
                result.append("%s/%s" % (path, c[0]))
            else:
                getAllNodesFromTypeOrNameList(tnlist[1:],
                                              c[2], "%s/%s" % (path, c[0]), result)
    return result


# --------------------------------------------------
def getAllNodesByTypeList(tree, typelist):
    return getPathsByTypeList(tree, typelist)


# --------------------------------------------------
def getPathsByTypeList(tree, typelist):
    """
    Returns a list of paths from the argument tree with nodes matching
    the list of types. The list you give is the list you would have if you
    pick the node type during the parse::

      tlist=['CGNSTree_t','CGNSBase_t','Zone_t']

      for path in getPathsByTypeList(T,tlist):
         node=getNodeByPath(T,path)
         # do something with node

    Would return all the zones of your tree.
    See also :py:func:`getPathsByTypeSet`

    :arg CGNS/Python tree: the start node of the CGNS tree to parse
    :arg list typelist: the (ordered) list of types
    :return:
      - a list of strings, each string is the path to a matching node
    """
    if tree[3] != typelist[0]:
        return []
    if tree[3] == CK.CGNSTree_ts:
        start = ""
    else:
        start = "%s" % tree[0]
    n = getAllNodesFromTypeList(typelist[1:], tree[2], start, [])
    return n


# --------------------------------------------------
def getAllNodesFromTypeList(typelist, node, path, result):
    for c in node:
        if c[3] == typelist[0]:
            if len(typelist) == 1:
                result.append("%s/%s" % (path, c[0]))
            else:
                getAllNodesFromTypeList(typelist[1:], c[2], "%s/%s" % (path, c[0]), result)
    return result


# --------------------------------------------------
def stackPath(path, *items):
    return stackPathItem(path, *items)


# --------------------------------------------------
def stackPathItem(path, *items):
    """
    Add the items to the path::

      p='/base'
      p1=stackPathItem(p,'Zone','FlowSolution')
      p2=stackPathItem(p,'ReferenceState')

    :arg str path: original path
    :arg *str items: tuple of strings to be de-referenced
    :return: a new path with concatenation of items as path tokens
    """
    pth = path
    for tk in items:
        pth += '/' + tk
    return getPathNormalize(pth)


# --------------------------------------------------
def getPaths(tree, path, plist):
    for c in tree[2]:
        if (len(c) > 1) and isinstance(c[0], str):
            plist.append(path + '/' + c[0])
            getPaths(c, path + '/' + c[0], plist)


# --------------------------------------------------
def getAllPaths(tree):
    plist = []
    path = ''
    getPaths(tree, path, plist)
    plist.sort()
    return plist


# def wsort(a, b):
#     if a[0] < b[0]:
#         return -1
#     if a[0] > b[0]:
#         return 1
#     if a[1] < b[1]:
#         return -1
#     if a[1] > b[1]:
#         return 1
#     return 0


# --------------------------------------------------
def getPathFullTree(tree, width=False):
    return getPathsFullTree(tree, width)


# --------------------------------------------------
def getPathsFullTree(tree, width=False, raw=False):
    """
    Returns the list of all possible node paths of a CGNS/Python tree::

      for path in getPathsFullTree(T):
          print path

    :arg CGNS/Python tree: target tree to parse
    :arb bool width: ``True`` for width sorting (default is ``False``)
    :arb bool raw: ``True`` to add top node (default is ``False``)
    :return:
      - A list of strings, each is a path
      - Empty list if tree is empty or invalid
    :Remarks:
      - The top node is ignored (usually /CGNSTree) unless you set ``raw``
      - When sorting with **width** the paths are listed as width-first parse
      - See also :py:func:`getPathListAsWidthFirstIndex`
    """
    if raw:
        tree = [None, None, [tree], None]
    r = getAllPaths(tree)
    if width:
        r.sort(key=lambda p: (p.count('/'), p))
    return r


# --------------------------------------------------
def checkPath(path, dienow=False):
    """
    Checks the compliance of a path, which is basically a UNIX-like
    path with constraints on each node name::

      checkPath('/Base/Zone/ZoneBC')

    :arg str path: path to check
    :return: ``True`` if the path is ok, ``False`` if a problem is found

    """
    if not isinstance(path, str) or not path:
        return False
    if path[0] == '/':
        path = path[1:]
    for p in path.split('/'):
        if not checkName(p, dienow):
            return False
    return True


# --------------------------------------------------
def hasSameRootPath(pathroot, pathtocompare):
    """
    Compares two paths::

      hasSameRootPath('/Base/Zone/ZoneBC','/Base/Zone/ZoneBC/BC#2/Data')
      # True
      hasSameRootPath('/Base/Zone/ZoneBC','/Base/ZoneBC#2')
      # False

    :arg str pathroot: root path to compare
    :arg str pathtocompare: path which is supposed to have rootpath as substring
    :return: True if 'rootpath' is a prefix of 'pathtocompare'
    :Remark:
      - Each node name is a token, see example below: the second example
        doesn't match as a path while it matches as a string.
    """
    l1 = getPathToList(pathroot)
    l2 = getPathToList(pathtocompare)
    if len(l1) > len(l2):
        return False
    for item1, item2 in zip(l1, l2):
        if item1 != item2:
            break
    else:
        return True
    return False


# --------------------------------------------------
def getPathListCommonAncestor(pathlist):
    """
    Finds the common ancestor for all paths in list::

      p=['/Base/Zone/Data-A','/Base/Zone/Data-D','/Base/Zone/ZoneBC/BC1']
      print getPathListCommonAncestor(p)
      # '/Base/Zone'

    :args list pathlist: list of path strings
    :return: The common root path (at least '/')
    """
    if len(pathlist) == 0:
        return '/'
    if len(pathlist) == 1:
        return pathlist[0]
    lp = []
    for p in pathlist:
        if p == '/':
            return '/'
        lp.append(getPathToList(p, True))
    t = lp[0]
    for p in lp:
        n = 0
        r = t
        m = min(len(p), len(r))
        for n in range(m):
            if p[n] != r[n]:
                t = r[:n]
                break
        else:
            t = r[:n + 1]
    if t:
        t = [''] + t
        c = '/'.join(t)
    else:
        c = '/'
    return c


# --------------------------------------------------
def getPathToList(path, nofirst=False, noroot=True):
    """
    Return the path as a list of node names::

      print getPathToList('/Base/Zone/ZoneBC')
      # ['','Base','Zone','ZoneBC']
      print getPathToList('/Base/Zone/ZoneBC',True)
      # ['Base','Zone','ZoneBC']
      print getPathToList('/')
      # []

    :arg str path: path string to split
    :arg bool nofirst: removes first empty string for absolute paths (default: False)
    :arg bool noroot: If true then removes the CGNS/HDF5 root if found (default: True)
    :return:
      - The list of path elements as strings
      - with '/' as argument, the function returns an empty list
    :Remarks:
      - The path is processed by :py:func:`getPathNormalize` before its split
    """
    lp = []
    if path is None:
        return []
    if len(path) > 0:
        path = getPathNormalize(path)
        if noroot:
            path = getPathNoRoot(path)
        if nofirst and (path[0] == '/'):
            path = path[1:]
        if path not in ['/', '']:
            lp = path.split('/')
    return lp


# --------------------------------------------------
def getPathAncestor(path, level=1, noroot=True):
    """
    Return the path of the node parent of the argument node path::

      print getPathAncestor('/Base/Zone/ZoneBC')
      # '/Base/Zone'

    :arg str path: string of the child node
    :arg int level: number of levels back from the child (default: 1 means the father of the node)
    :return:
      - The ancestor path
      - If the path is '/' its ancestor is None.
    """
    lp = getPathToList(path, noroot=noroot)
    if (len(lp) == 2) and (lp[0] == ''):
        ancestor = '/'
    elif len(lp) > 1:
        ancestor = '/'.join(lp[:-1])
    elif len(lp) == 1:
        ancestor = '/'
    else:
        ancestor = None
    if level == 0:
        ancestor = path
    if level > 1:
        ancestor = getPathAncestor(ancestor, level - 1)
    return ancestor


# --------------------------------------------------
def getDepth(path):
    return len(getPathToList(path, True))


# --------------------------------------------------
def getPathLeaf(path):
    """
    Return the leaf node name of the path::

      print getPathLeaf('/Base/Zone/ZoneBC')
      # 'ZoneBC'

    :arg str path: a CGNS/Python path
    :return:
      - The leaf node name (the last token of the path)
      - If the path is '/' the function returns '' (empty string)

    """
    leaf = ''
    lp = getPathToList(path)
    if len(lp) > 0:
        leaf = lp[-1]
    return leaf


# --------------------------------------------------
def getPathNoRoot(path):
    """
    Return an absolute path without the implementation nodes 'HDF5 Mother node'
    or 'CGNSTree' if detected as first element::

      print getPathNoRoot('/HDF5 Mother Node/Base/Zone/ZoneBC')
      # ['Base','Zone','ZoneBC']

    :arg str path: the path of the node
    :return: The new path without root implementation node if found
    :Remarks:
      - a '/' is expected as first char, it is forced if not found
      - The path is processed by :py:func:`getPathNormalize`
      - Implementation root can be CGNS.PAT.cgnskeywords.CGNSHDF5ROOT_s as
        well as CGNS.PAT.cgnskeywords.CGNSTree_s
    """
    path = getPathNormalize(path)
    if path in [None, "", ".", '/', "/" + CK.CGNSHDF5ROOT_s, "/" + CK.CGNSTree_s]:
        return "/"
    if path[0] != '/':
        path = '/' + path
    lp = path.split('/')
    if lp[0] in [CK.CGNSHDF5ROOT_s, CK.CGNSTree_s]:
        lp = lp[1:]
    if ((lp[0] == '') and
            (len(lp) > 1) and
            (lp[1] in [CK.CGNSHDF5ROOT_s, CK.CGNSTree_s])):
        lp = [lp[0]] + lp[2:]
    path = '/'.join(lp)
    if (path == '') or (path[0] != '/'):
        path = '/' + path
    return path


# --------------------------------------------------
def getPathAsTypes(tree, path, legacy=True):
    """Return the list of types corresponding to the argument path in the tree::

      getPathAsTypes(T,'/Base/Zone/ZoneBC')
      # ['CGNSBase_t','Zone_t','ZoneBC_t']

    :arg CGNS/Python tree: target tree
    :arg str path: path to parse get
    :return:
      - The list of CGNS/SIDS types found (as strings)
      - `None` if the path is not found
    """
    ltypes = []
    if checkRootNode(tree, legacy=False):
        p = getPathToList(path, noroot=False, nofirst=True)
        if p and (p[0] != CK.CGNSTree_s):
            path = '/' + CK.CGNSTree_s + '/' + path
            legacy = False
    path = getPathNormalize(path)
    while path not in ['/', None]:
        t = getTypeByPath(tree, path)
        ltypes += [t]
        path = getPathAncestor(path, noroot=False)
    if legacy and (len(ltypes) > 0):
        ltypes = ltypes[:-1]
    ltypes.reverse()
    return ltypes


# --------------------------------------------------
def getPathNormalize(path):
    """Return the same path as minimal string, removes `////` and `/./` and
    other simplifiable UNIX-like path elements::

        # a checkPath here would fail, because single or double dots are not
        # allowed as a node name. But actually this is allowed as a
        # path description
        p=getPathNormalize('///Base/././//Zone/../Zone/./ZoneBC//.')

        # would return '/Base/Zone/ZoneBC'
        if (not checkPath(p)):
            print 'something bad happens'

    :arg str path: the path of the node
    :return: The simplified path
    :Remarks:
      - Uses *os.path.normpath* and replaces \\ if windows os.path
      - Before its normalization a path can be **non-compliant**
    """
    if path == '':
        return ''
    if path is None:
        return '/'
    first = ((len(path) > 0) and (path[0] == '/'))
    result_tokens = []
    tokens = path.split('/')
    for token in tokens:
        if token in ['', '.']:
            continue
        if result_tokens and (token == '..'):
            result_tokens.pop()
        else:
            result_tokens.append(token)
    path = '/'.join(result_tokens)
    if first:
        path = '/' + path
    if path == '':
        path = '.'
    return path


# --------------------------------------------------
def childNames(node):
    return childrenNames(node)


def childrenNames(node):
    """Gets the children names as a list of strings::

      for c in childrenNames(node):
          print '%s/%s'%(node[0],c)

    :arg CGNS/Python node: the parent node
    :return: List of children names (str)
    """
    r = []
    if node is None:
        return r
    for c in node[2]:
        r.append(c[0])
    return r


# --------------------------------------------------
def getAllNodesByTypeSet2(typelist, tree):
    if tree[3] == CK.CGNSTree_ts:
        start = "/%s" % tree[0]
    else:
        start = "%s" % tree[0]
    n = getAllNodesFromTypeSet(typelist, tree[2], start, [])
    return n


# --------------------------------------------------
def getAllNodesByTypeSet(tree, typeset):
    return getPathsByTypeSet(tree, typeset)


# --------------------------------------------------
def getPathsByNameSet(tree, nameset):
    """
    Returns a list of paths from the argument tree with nodes matching
    one of the names in the list::

      #  Would return all the nodes with names *BCWall* or  *BCExt*
      tset=['BCWall','BCExt']

      for path in getPathsByNameSet(T,tset):
          node=getNodeByPath(T,path)
          # do something

    :arg CGNS/Python tree: start node of the CGNS tree to parse
    :arg list nameset: the list of names
    :return: a list of strings, each string is the path to a matching node
    :Remarks: See also :py:func:`getPathsByTypeSet`

    """
    if tree[3] == CK.CGNSTree_ts:
        start = ""
    else:
        start = "%s" % tree[0]
    n = getAllNodesFromNameSet(nameset, tree[2], start, [])
    return n


# --------------------------------------------------
def getPathsByTypeSet(tree, typeset):
    """
    Returns a list of paths from the argument tree with nodes matching
    one of the types in the list::

      #  Would return all the zones and BCs of your tree.
      tset=['BC_t','Zone_t']

      for path in getPathsByTypeSet(T,tset):
          node=getNodeByPath(T,path)
          # do something

    :arg CGNS/Python tree: start node of the CGNS tree to parse
    :arg list typeset: the list of CGNS/SIDS types as strings
    :return: a list of strings, each string is the path to a matching node
    :Remarks: See also :py:func:`getPathsByTypeList`

    """
    if tree[3] == CK.CGNSTree_ts:
        start = ""
    else:
        start = "%s" % tree[0]
    n = getAllNodesFromTypeSet(typeset, tree[2], start, [])
    return n


# --------------------------------------------------
def getPathListAsWidthFirstIndex(paths, fileindex=1):
    """
    The order of the paths for a given depth is the alphabetical order of
    the full path to the node. The width parse goes through all the children
    of a given depth, for all parents.

    For example, you want to loop on a list of nodes you retrieved from
    another function call. You want to make sure that your loop actually
    follows a width-first constraint for parse purpose. You first call
    `getPathListAsWidthFirstIndex` to get the list in the right order.

    As the returned value also contains the index of the path, you can perform
    you simple loop by getting only the path::

      listpath=someFonctionReturnsPathList( ... )
      sortedlistpath=[s[2] for s in getPathListAsWidthFirstIndex(lispath)]

    :arg list paths: list of paths to order
    :arg int fileindex: index of the current "file" (default is 1)
    :return: ordered list with the pattern [ [<int:file-index>, <int:child-index>, <string:path>] ... ]
    :Remarks:
      - Children index goes from 0 to N-1
    """
    dpth = {}
    for p in paths:
        d = len(p.split('/'))
        if d not in dpth:
            dpth[d] = []
        dpth[d].append(p)
    for d in dpth:
        dpth[d].sort()
    k = list(dpth)
    k.sort()
    count = 0
    ix = []
    for d in k:
        for p in dpth[d]:
            ix.append([fileindex, count, p])
            count += 1
    return ix


# --------------------------------------------------
def getAllNodesAsWidthFirstIndex(tree, fileindex=1):
    lpth = getAllPaths(tree)
    return getPathListAsWidthFirstIndex(lpth)


# --------------------------------------------------
def getAllNodesFromNameSet(namelist, node, path, result):
    for c in node:
        if c[0] in namelist:
            result.append("%s/%s" % (path, c[0]))
        getAllNodesFromNameSet(namelist, c[2], "%s/%s" % (path, c[0]), result)
    return result


# --------------------------------------------------
def getAllNodesFromTypeSet(typelist, node, path, result):
    for c in node:
        if c[3] in typelist:
            result.append("%s/%s" % (path, c[0]))
        getAllNodesFromTypeSet(typelist, c[2], "%s/%s" % (path, c[0]), result)
    return result


# --------------------------------------------------
def getNodeAllowedChildrenTypes(pnode, node):
    """
    Returns all allowed CGNS-types for the node. The parent is mandatory::
       if (node[2] not in getNodeAllowedChildrenTypes(parent,node)):
           print 'Such a child is not SIDS compliant'
    :arg CGNS/Python pnode: parent node of second argument
    :arg CGNS/Python node: target node
    :return: A list of CGNS/SIDS types (strings)
    :Remarks:
      - The parent node is mandatory, many CGNS/SIDS types are allowed in many
        places and the only way to check their compliance is to have their
        father node.
    """
    tlist = []
    if node[3] == CK.CGNSTree_ts:
        return tlist
    try:
        if (node[3] is None) or (pnode is None):
            ctl = CT.types[CK.CGNSTree_ts]
        else:
            ctl = CT.types[pnode[3]]
        for cn in ctl.children:
            if cn[0] not in tlist:
                tlist += [cn[0]]
    except:
        pass
    return tlist


# --------------------------------------------------
def getNodeAllowedDataTypes(node):
    """Returns a list of string with all allowed CGNS data types for the node::
         node=['ReferenceState',numpy.array((1,2,3)),[],'ReferenceState_t']
         if (getValueDataType(node) not in getNodeAllowedDataTypes(node)):
              print 'Node %s has bad value type'%(node[0])
    :arg CGNS/Python node: target node
    :return:
      - A list of CGNS/SIDS value data types (strings)
      - see also :py:func:`getValueDataType`
    """
    tlist = []
    try:
        tlist = CT.types[node[3]].datatype
    except:
        pass
    return tlist


# --------------------------------------------------
def getAllFamilies(tree):
    """Return all the Family_t of the tree (all CGNSBases)"""
    fpth = [CK.CGNSTree_ts, CK.CGNSBase_ts, CK.Family_ts]
    famlist = getAllNodesByTypeOrNameList(tree, fpth)
    return [getPathLeaf(f) for f in famlist]


# --------------------------------------------------
def getZoneFromFamily(tree, families, additional=True):
    """Return the Zone paths for all Zones having the target families as
    FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg list str families: list of family names to look for
    :arg boolean additional: also looks for AdditionalFamilyName_t (default True)
    :return: list of Zone paths
    """
    fpth1 = [CK.CGNSTree_ts, CK.CGNSBase_ts, CK.Zone_ts, CK.FamilyName_ts]
    fpth2 = [CK.CGNSTree_ts, CK.CGNSBase_ts, CK.Zone_ts, CK.AdditionalFamilyName_ts]
    zlist = getAllNodesByTypeOrNameList(tree, fpth1)
    if additional:
        zlist += getAllNodesByTypeOrNameList(tree, fpth2)
    r = []
    for pth in zlist:
        if getValueByPath(tree, pth).toString() in families:
            r += [getPathAncestor(pth)]
    return r


# --------------------------------------------------
def getBCFromFamily(tree, families, additional=True):
    """Return the BC paths for all BCs having the target families as
    FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg list str families: list of family names to look for
    :arg boolean additional: also looks for AdditionalFamilyName_t (default True)
    :return: list of BC paths
    """
    fpth0 = [CK.CGNSTree_ts, CK.CGNSBase_ts, CK.Zone_ts, CK.ZoneBC_ts, CK.BC_ts]
    fpth1 = fpth0 + [CK.FamilyName_ts]
    fpth2 = fpth0 + [CK.AdditionalFamilyName_ts]
    zlist = getAllNodesByTypeOrNameList(tree, fpth1)
    if additional:
        zlist += getAllNodesByTypeOrNameList(tree, fpth2)
    r = []
    for pth in zlist:
        if getValueByPath(tree, pth).tostring().decode('ascii') in families:
            r += [getPathAncestor(pth)]
    return r


# --------------------------------------------------
def getZoneSubRegionFromFamily(tree, families, additional=True):
    """Return the ZoneSubRegion paths for all having the target families as
    FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg list str families: list of family names to look for
    :arg boolean additional: also looks for AdditionalFamilyName_t (default True)
    :return: list of ZoneSubRegion paths
    """
    fpth0 = [CK.CGNSTree_ts, CK.CGNSBase_ts, CK.Zone_ts, CK.ZoneBC_ts,
             CK.ZoneSubRegion_ts]
    fpth1 = fpth0 + [CK.FamilyName_ts]
    fpth2 = fpth0 + [CK.AdditionalFamilyName_ts]
    zlist = getAllNodesByTypeOrNameList(tree, fpth1)
    if additional:
        zlist += getAllNodesByTypeOrNameList(tree, fpth2)
    r = []
    for pth in zlist:
        if getValueByPath(tree, pth).tostring().decode('ascii') in families:
            r += [getPathAncestor(pth)]
    return r


# --------------------------------------------------
def getFamiliesFromZone(tree, zonepath):
    """Return all the Zone's FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg str zonepath: target zone
    :return: list of family names
    """
    return getFamiliesFromBC(tree, zonepath)


# --------------------------------------------------
def getFamiliesFromBC(tree, path):
    """Return all the BC's FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg str path: target path
    :return: list of family names
    """
    node = getNodeByPath(tree, path)
    l1 = hasChildType(node, CK.FamilyName_ts)
    l2 = hasChildType(node, CK.AdditionalFamilyName_ts)
    if l1 is None:
        l1 = []
    if l2 is None:
        l2 = []
    r = []
    for nd in l1 + l2:
        r.append(nd[1].tostring().decode('ascii'))
    return r


# --------------------------------------------------
def getFamiliesFromZoneSubRegion(tree, zonepath):
    """Return all the ZoneSubRegion's FamilyName_t or AdditionalFamilyName_t.

    :arg CGNS/Python tree: target tree to parse
    :arg str zonepath: target zone
    :return: list of family names
    """
    return getFamiliesFromBC(tree, zonepath)


# -----------------------------------------------------------------------------
def hasChildType(parent, ntype):
    """Checks if the parent node has a child with given type::

       node=getNodeByPath(T,'/Base/Zone/BC')
       nl=hasChildType(node, 'AdditionalFamily_t')
       for n in nl:
         v=getValue(n)

    :arg CGNS/Python parent: target node
    :arg str ntype: CGNS/SIDS node type to look for
    :return: List of nodes with this type (can be empty)

    """
    if not parent:
        return []
    r = []
    for n in parent[2]:
        if n[3] == ntype:
            r.append(n)
    if r:
        return r
    return []


# -----------------------------------------------------------------------------
def getEnumAsString(node):
    """
    Return the node value as corresponding SIDS enumerate string::
        if (hasEnumValue(node)):
            print getEnumAsString(node)

    The enumerate value is the <node-value>.flat[0] that is supposed to be an
    integer.

    :arg CGNS/Python node: target node
    :return: A string corresponding to the SIDS enumerate value
    :Remarks:
      - returns empty string if something wrong happens
      - See also :py:func:`hasEnumValue`
    """
    try:
        stp = node[3]
        if stp == CK.Elements_ts:
            stp = CK.ElementType_ts
        e = CK.cgnsenums[stp]
        v = e[node[1].flat[0]]
        return v
    except:
        return ''


# -----------------------------------------------------------------------------
def hasEnumValue(node):
    """
    Checks if the node type allows a SIDS enum value::
        if (hasEnumValue(node)):
            print getEnumAsString(node)
    :arg CGNS/Python node: target node
    :return: True if the node value is a SIDS enumerate
    :Remarks:
      - See also :py:func:`getEnumAsString`
    """
    stp = node[3]
    if stp == CK.Elements_ts:
        stp = CK.ElementType_ts
    if stp in CK.cgnsenums:
        return True
    return False


# -----------------------------------------------------------------------------
def hasChildName(parent, name, dienow=False):
    """
    Finds a child with a target name. Returns the node.
    """
    return hasChildNode(parent, name, dienow)


# -----------------------------------------------------------------------------
def setChildName(parent, oldname, newname, dienow=False):
    """
    Changes the name of an existing node::

        n=hasChildNode(zone,CGK.ZoneType_s)
        for nc in childrenNames
            nc=setChildName(parent,zonename,zonename+'#01')

    :arg CGNS/Python parent: the parent node
    :arg str oldname: the child name to look for
    :arg str newname: the new name
    :return:
      - parent node
    :Remarks:
      - uses :py:func:`hasChildName` to check if old name exists and new name
        does not.
    """
    n1 = hasChildName(parent, oldname)
    n2 = hasChildName(parent, newname)
    if n1 and not n2:
        n1[0] = newname
    return parent


# -----------------------------------------------------------------------------
def hasChildNode(parent, name, dienow=False):
    """
    Returns a child node if it exists::

      n=hasChildNode(zone,CGK.ZoneType_s)
      if ((n is not None) and (stringValueMatches(n,CGK.Unstructured_s)):
          # found unstructured zone

    :arg CGNS/Python parent: the parent node
    :arg str name: the child name to look for
    :return:
      - the actual child node if the child exists
      - None if the child is not found
    :raises: :ref:`cgnsnameerror` code 102 if `dienow` is True
    """
    if not parent:
        return None
    if parent[2] is None:
        return None
    for nc in parent[2]:
        if nc[0] == name:
            if dienow:
                raise CE.cgnsNameError(102, (name, parent[0]))
            return nc
    return None


# --------------------------------------------------
def getTypeAsGrammarToken(ntype):
    if ntype in CK.weirdSIDStypes:
        return CK.weirdSIDStypes[ntype]
    return ntype


# --------------------------------------------------
def hasChildNodeOfType(node, ntype):
    if node is None:
        return 0
    for cn in node[2]:
        if cn[3] == ntype:
            return 1
    return 0


# --------------------------------------------------
def stringMatches(string, reval):
    """Returns re.group if the pattern matches the string, None otherwise"""
    return re.match(reval, string)


# --------------------------------------------------
def stringNameMatches(node, reval):
    """Returns re.group if the pattern matches the node name, None otherwise"""
    return stringMatches(node[0], reval)


# --------------------------------------------------
def stringTypeMatches(node, reval):
    """Returns re.group if the pattern matches the node type, None otherwise"""
    return stringMatches(node[3], reval)


# --------------------------------------------------
def stringValueMatches(node, reval):
    """True if the string matches the node value"""
    if node is None:
        return False
    if node[1] is None:
        return False
    if getNodeType(node) != CK.C1:
        return False
    if isinstance(node[1], str):
        vn = node[1]
    elif isinstance(node[1], numpy.ndarray) and (node[1].dtype.kind in ['S', 'a']):
        vn = node[1].tostring().decode('ascii', 'strict')
    else:
        return False
    if stringMatches(vn, reval) is not None:
        return True
    return False


# --------------------------------------------------
def stringValueInList(node, listval):
    """True if the value list contains the node value"""
    if node is None:
        return False
    if node[1] is None:
        return False
    if getNodeType(node) != CK.C1:
        return False

    if isinstance(node[1], str):
        vn = node[1]
    elif isinstance(node[1], numpy.ndarray) and (node[1].dtype.kind in ['S', 's']):
        vn = node[1].tostring().decode('ascii', 'strict')
    else:
        return False
    return vn in listval


# --------------------------------------------------
def isLinkNode(tree, links, node):
    # """Returns True if the node is a linked node"""
    pass


# --------------------------------------------------
def checkLinkFile(lkfile, lksearch=None):
    """Checks if the file exists"""
    found = (None, None)
    if not lksearch:
        lksearch = ['']  # None, []
    for spath in lksearch:
        sfile = op.normpath(spath + '/' + lkfile)
        if op.exists(sfile):
            found = (op.normpath(spath), op.normpath(lkfile))
            break
    return found


# --------------------------------------------------
def copyArray(a):
    """Copy a numpy.ndarray with flags"""
    if not isinstance(a, numpy.ndarray):
        return None  # None, []
    if numpy.isfortran(a):
        b = numpy.array(a, order='F', copy=True)
    else:
        b = numpy.array(a, copy=True)
    return b


# --------------------------------------------------
def getChildren(t):
    s = set([c[0] for c in t[2]])
    cl = {}
    for c in t[2]:
        cl[c[0]] = c
    return s, cl


# --------------------------------------------------
# should merge with checkSameValue that returns True/False
#
def compareValues(na, nb):
    va = na[1]
    vb = nb[1]
    eps = 1e-12
    if va is None and vb is not None:
        return 1
    if va is not None and vb is None:
        return 2
    if va is None and vb is None:
        return 0
    if (not isinstance(va, numpy.ndarray) or
            not isinstance(vb, numpy.ndarray)):
        return 0
    if va.dtype.char != vb.dtype.char:
        return 3
    if va.size != vb.size:
        return 4
    if va.shape != vb.shape:
        return 5
    if va.dtype.char not in ['c', 'S']:
        vd = numpy.array(numpy.abs(va - vb))
        if numpy.any(vd > eps):
            return 6
    elif va.ravel().tostring() != vb.ravel().tostring():
        return 7
    return 0


# --------------------------------------------------
def diff(ta, tb, path='', tag='A', diag=None, trace=False):
    if diag is None:
        diag = {}
    diag[path] = []
    if ta[3] != tb[3]:
        diag[path].append(('CT',))
        if trace:
            print('CT %s %s' % (tag, path))
    dta = getValueDataType(ta)
    dtb = getValueDataType(tb)
    # dta = CGNS.PAT.cgnsutils.getValueDataType(ta)
    # dtb = CGNS.PAT.cgnsutils.getValueDataType(tb)
    #
    if dta is not CGNS.PAT.cgnskeywords.MT:
        dnum = compareValues(ta, tb)
        if dnum:
            diag[path].append(('C%d' % dnum,))
            if trace:
                print('C%d %s %s' % (dnum, tag, path))
    (sa, da) = getChildren(ta)
    (sb, db) = getChildren(tb)
    a2b = sa.difference(sb)
    sn = sa.union(sb)
    for cn in sn:
        np = path + '/' + cn
        a = cn in sa
        b = cn in sb
        if not a and b:
            diag[path].append(('NA', np))
            if trace:
                print('NA %s %s' % (tag, np))
        if a and not b:
            diag[path].append(('ND', np))
            if trace:
                print('ND %s %s' % (tag, np))
        if a and b:
            diff(da[cn], db[cn], np, tag, diag, trace)
    if not diag[path]:
        del diag[path]
    return diag


# --------------------------------------------------
def diffAnalysis(diag):
    s = '## ++ A node not in B\n'
    s += '## -- B node not in A\n'
    s += '## >t A and B node SIDS type differ\n'
    s += '## >a A and B node values differ (one has None)\n'
    s += '## >x A and B node values differ (not same data type)\n'
    s += '## >n A and B node values differ (not same total size)\n'
    s += '## >d A and B node values differ (not same shape)\n'
    s += '## >c A and B node values differ (not same contents)\n'
    s += '## >s A and B node values differ (string values)\n'
    paths = list(diag)
    paths.sort()
    for k in paths:
        for d in diag[k]:
            ldiag = ''
            if d[0] == 'NA':
                ldiag = '++'
            if d[0] == 'ND':
                ldiag = '--'
            if d[0] in ['CT']:
                ldiag = '>t'
            if d[0] in ['C1', 'C2']:
                ldiag = '>a'
            if d[0] in ['C3']:
                ldiag = '>x'
            if d[0] in ['C6']:
                ldiag = '>c'
            if d[0] in ['C7']:
                ldiag = '>s'
            if d[0] in ['C4']:
                ldiag = '>n'
            if d[0] in ['C5']:
                ldiag = '>d'
            if len(d) > 1:
                s += '%s %s\n' % (ldiag, d[1])
            else:
                s += '%s %s\n' % (ldiag, k)
    return s


# --------------------------------------------------
def toStringValue(v):
    """ASCII pretty print of one node value."""
    if v is None:
        return None
    ao = 'C'
    if numpy.isfortran(v):
        ao = 'F'
    at = v.dtype.name
    av = v.tolist()
    return "numpy.array(%s,dtype='%s',order='%s')" % (av, at, ao)


# --------------------------------------------------
def toStringChildren(l, readable=False, shift=0, keywords=False, pycgns=False):
    """ASCII pretty print of one children node list."""
    s = "["
    for c in l:
        s += toString(c, readable, shift, keywords)
        s += ','
    if l:
        s = s[:-1]
    if readable and l:
        s += '],\n' + (shift - 1) * ' '
    else:
        s += '],'
    return s


# --------------------------------------------------
def prettyPrint(tree, path='', depth=0, sort=False, links=None, paths=None):
    """ASCII pretty print of the whole tree."""
    if links is None:
        links = []
    if paths is None:
        paths = []
    print(depth * ' ',)
    path = path + '/' + tree[0]
    pnr = getPathNoRoot(path)
    nt = getNodeType(tree)
    ns = getShape(tree)
    if paths:
        d = {p[0]: p[2:] for p in paths}
        if pnr in d:
            nt = d[pnr][0]
            ns = d[pnr][1]
    n = "%s(%s) %s:%s" % (tree[0], tree[3], nt, ns),
    print("%-32s" % n)
    if not sort:
        for c in tree[2]:
            prettyPrint(c, path, depth=depth + 2, links=links, paths=paths)
    else:
        for c in getNextChildSortByType(tree, criteria=CK.sortedtypelist):
            prettyPrint(c, path, depth=depth + 2, sort=True, links=links, paths=paths)


# --------------------------------------------------
def toString(tree, readable=False, shift=0, keywords=False, pycgns=False):
    """
    Returns the full sub-tree as a single line string::

      s=toString(tree)
      print s

    :arg CGNS/Python tree: the CGNS/Python tree to parse
    :return: A string representation of the whole tree
    :Remarks:
      - the `numpy` module is used, the values strings are actual string
        representation of a `numpy` array, include `dtype` and `order`

    """
    if pycgns:
        keywords = True
    n = tree
    prefix = ''
    if pycgns and (shift == 0):
        prefix += """# ---
import numpy
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU

tree=\\\n"""
    if readable and shift:
        prefix = '\n' + shift * ' '
    if n is None:
        return 'None'
    name = n[0]
    if keywords and name in CK.cgnsnames:
        name = 'CGK.%s_s' % n[0]
    else:
        name = "'%s'" % n[0]
    s = "%s[%s,%s,%s" % (prefix,
                         name,
                         toStringValue(n[1]),
                         toStringChildren(n[2], readable, shift + 2, keywords, pycgns))
    if keywords:
        s += 'CGK.%ss ]' % n[3]
    else:
        s += "'%s' ]" % n[3]
    return s


# --------------------------------------------------
def toFile(tree, filename):
    """
    Writes the CGNS/Python tree as an ``import``-able string in a file::

      tofile(tree,'NACA0012.py')
      import NACA0012 as M
      T=M.T

    The actual node variable is ``T``, we retrieve this value from the module
    ``M``, which is the result of the import.

    :arg CGNS/Python tree: the tree structure to save
    :arg str filename: the name of the file
    :Remarks:
      - you have to use the``.py`` file extension if you use usual imports
      - calls :py:func:`toString`

    """
    f = open(filename, 'w+')
    f.write('import numpy\nT=')
    f.write(toString(tree, readable=True))
    f.close()


# --------------------------------------------------
def getNodesFromTypeSet(tree, typeset):
    """  **Iterator** returns the children nodes of the argument CGNS/Python
    matching one of the type in the list. It allows an alternate traversal
    of the CGNS/Python tree::

      for path, child in getNodesFromTypeSet(tree, [CGK.Zone_ts]):
          print 'Next child:', child[0]

    :arg CGNS/Python tree: the tree structure to traverse
    :arg list typeset: the list of CGNS/SIDS types as strings
    :return:
      - This is an iterator, it returns a tuple (path, CGNS/Python node)
    :Remarks:
      - this iterator must be used carefully to scan a CGNS/Python tree
        or do inplace modification. Modification of tree structure while
        using this iterator is hazardous.
    """

    def genNodesFromTypeSet(typelist, cnode, cpath):
        for c in cnode:
            if c[3] in typelist:
                yield ("%s/%s" % (cpath, c[0]), c)
            for curpath, curnode in genNodesFromTypeSet(typelist, c[2], "%s/%s" % (cpath, c[0])):
                yield (curpath, curnode)

    if tree[3] == CK.CGNSTree_ts:
        start = ""
    else:
        start = "%s" % tree[0]
    for path, node in genNodesFromTypeSet(typeset, tree[2], start):
        yield (path, node)

# ----
