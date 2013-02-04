.. -------------------------------------------------------------------------
.. pyCGNS.PAT - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.PAT
========

The *PATtern* module provides the user with functions dedicated to
:term:`CGNS/Python` trees.
The :ref:`PAT.cgnslib <pat_cgnslib>` module uses the *SIDS* compliant data 
structures, you
can create, read, check, modify some 
:term:`CGNS/Python` sub-trees related to a
*SIDS* type.
With this module you are working with a Python data structure, all function
are using plain Python/Numpy objects. Thus, the  *PAT* module is not required
for your applications, as you can write your own function to handle these
Python objects.
The :ref:`PAT.cgnsutils <pat_cgnsutils>` provides utility fonctions 
for raw :term:`CGNS/Python` trees or nodes.
The *PAT* defines also constant modules such as 
:ref:`PAT.cgnskeywords <pat_cgnskeywords>` for all
*SIDS* names or constant strings, 
:ref:`PAT.cgnstypes <pat_cgnstypes>` for the *SIDS* types
descriptions (enumerates, allowed list of children...) and the 
:ref:`PAT.cgnserrors <pat_cgnserrors>`
with error codes and their messages.

A special module :ref:`PAT.SIDS <sids_patterns>` has all *CGNS/SIDS* 
patterns gathered as 
*PAT.cgnslib* calls. These patterns, used for creation only, are building 
in a recursive way the whole sub-tree for a given *SIDS* type.

.. _pat_cgnsutils:

Utilities
---------

The `CGNS.PAT.cgnsutils` has a large set of utility functions using
the :term:`CGNS/Python` nodes,
sub-trees or trees as arguments, you can manipulate tree paths, links,
values.  Functions are not gathered into a class because we want them
to proceed on standard :term:`CGNS/Python` trees. Most functions have an
optional error management, you can ask them to raise an exception or
to return None. The `dienow` argument is set to `False` as default,
which means a error would return a `None`. A `dienow` set to `True`
raises an :ref:`error <pat_cgnserrors>`.
Some functions also have an
optional legacy management, to take into account old CGNS/Python stuff.
When set to `True`, the `CGNSTree_t` top node should not appear and is
not inserted when needed. The weird CGNS/SIDS types such as `"int[IndexDimension]"`
are used instead of CGNS/Python replacements.
The `legacy` argument is set to `False` as default.

The list below gives an overview of publicly available functions.

+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Node Life Cycle**                                                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Create a new node                                           | :py:func:`nodeCreate <CGNS.PAT.cgnsutils.nodeCreate>`                                   |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Deep copy of node                                           | :py:func:`nodeCopy <CGNS.PAT.cgnsutils.nodeCopy>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Delete node and its children                                | :py:func:`nodeDelete <CGNS.PAT.cgnsutils.nodeDelete>`                                   |                 
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Node structure and contents**                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks basic node structure                                 | :py:func:`checkNode <CGNS.PAT.cgnsutils.checkNode>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node is a CGNS/Python tree root         | :py:func:`checkRootNode <CGNS.PAT.cgnsutils.checkRootNode>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node is of arg CGNS/SIDS type           | :py:func:`checkNodeType <CGNS.PAT.cgnsutils.checkNodeType>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the node name has CGNS/SIDS correct syntax        | :py:func:`checkNodeName <CGNS.PAT.cgnsutils.checkNodeName>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if two nodes have same contents                        | :py:func:`checkSameNode <CGNS.PAT.cgnsutils.checkSameNode>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if arg name is not already in the parent children list | :py:func:`checkDuplicatedName <CGNS.PAT.cgnsutils.checkDuplicatedName>`                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if arg path is CGNS/SIDS compliant                   | :py:func:`checkPath <CGNS.PAT.cgnsutils.checkPath>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+               
|                                                             | **Boolean tests**                                                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if node has a child of arg type                        | :py:func:`hasChildType <CGNS.PAT.cgnsutils.hasChildType>`                               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if any ancestor of node has arg type                   | :py:func:`hasAncestorType <CGNS.PAT.cgnsutils.hasAncestorType>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if node has child with arg name                        | :py:func:`hasChildName <CGNS.PAT.cgnsutils.hasChildName>`                               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if any ancestor of node has arg name                   | :py:func:`hasAncestorName <CGNS.PAT.cgnsutils.hasAncestorName>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if node value matches                                  | :py:func:`hasValue <CGNS.PAT.cgnsutils.hasValue>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if node value datatype matches                         | :py:func:`hasValueDataType <CGNS.PAT.cgnsutils.hasValueDataType>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if node value has flags (Numpy flags)                  | :py:func:`hasValueFlags <CGNS.PAT.cgnsutils.hasValueFlags>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+ 
|                                                             | **Search and Retrieve**                                                                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns a node from its path                                | :py:func:`getNodeByPath <CGNS.PAT.cgnsutils.getNodeByPath>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns a node value from node path                         | :py:func:`getValueByPath <CGNS.PAT.cgnsutils.getValueByPath>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns list of node children from path                     | :py:func:`getChildrenByPath <CGNS.PAT.cgnsutils.getChildrenByPath>`                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns a node CGNS/SIDS type from path                     | :py:func:`getTypeByPath <CGNS.PAT.cgnsutils.getTypeByPath>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| All nodes matching the set of CGNS/SIDS types               | :py:func:`getAllNodesByTypeSet <CGNS.PAT.cgnsutils.getAllNodesByTypeSet>`               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+             
|                                                             | **CGNS/SIDS information**                                                               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of allowed CGNS/SIDS types as child of node            | :py:func:`getNodeAllowedChildrenTypes <CGNS.PAT.cgnsutils.getNodeAllowedChildrenTypes>` |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of allowed CGNS/SIDS data types for this node          | :py:func:`getNodeAllowedDataTypes <CGNS.PAT.cgnsutils.getNodeAllowedDataTypes>`         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Value manipulation**                                                                  |               
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns node value shape (numpy.ndarray)                    | :py:func:`getValueShape <CGNS.PAT.cgnsutils.getValueShape>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns node value data type as CGNS/SIDS data type         | :py:func:`getValueDataType <CGNS.PAT.cgnsutils.getValueDataType>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node has a value                        | :py:func:`hasValue <CGNS.PAT.cgnsutils.hasValue>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node value data type is correct         | :py:func:`hasValueDataType <CGNS.PAT.cgnsutils.hasValueDataType>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Path functions**                                                                      |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the path of a node                                      | :py:func:`getPathFromNode <CGNS.PAT.cgnsutils.getPathFromNode>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get list of all CGNS/Python tree paths                      | :py:func:`getPathFullTree <CGNS.PAT.cgnsutils.getPathFullTree>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get list of all tree paths with name filter                 | :py:func:`getPathByNameFilter <CGNS.PAT.cgnsutils.getPathByNameFilter>`                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get list of all tree paths with type filter                 | :py:func:`getPathByTypeFilter <CGNS.PAT.cgnsutils.getPathByTypeFilter>`                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns the path string as a list of node names             | :py:func:`getPathToList <CGNS.PAT.cgnsutils.getPathToList>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the parent path of the arg path                         | :py:func:`getPathAncestor <CGNS.PAT.cgnsutils.getPathAncestor>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the last node name of the arg path                      | :py:func:`getPathLeaf <CGNS.PAT.cgnsutils.getPathLeaf>`                                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get path with CGNS/Python root removed                      | :py:func:`getPathNoRoot <CGNS.PAT.cgnsutils.getPathNoRoot>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get list of types of nodes along the arg path               | :py:func:`getPathAsTypes <CGNS.PAT.cgnsutils.getPathAsTypes>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Normalizes the arg path                                     | :py:func:`getPathNormalize <CGNS.PAT.cgnsutils.getPathNormalize>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+

.. automodule:: CGNS.PAT.cgnsutils
   :members:

.. _pat_cgnslib:

The Pythonish CGNS lib
----------------------

The so-called `CGNSlib` or `MLL` or `Mid-level` library, is set of functions
for used to read/write/modify a set of nodes matching a CGNS/SIDS type.
The Pythonish flavour of this library declares a set of functions with more
or less the same interface but with Python values.

.. automodule:: CGNS.PAT.cgnslib
   :members:

.. _sids_patterns:

SIDS patterns
-------------

The patterns are importable modules, they create a complete *SIDS* sub-tree
with default values. There is no way to customize the default values or the
actual contents of the sub-tree. The pattern creates the mandatory as well
as the optional nodes. Once created, the user has to modify the sub-tree
using the :ref:`PAT.cgnsutils <pat_cgnsutils>` 
or :ref:`PAT.cgnslib <pat_cgnslib>` functions.

Once the pattern module is imported, the actual pattern is  referenced by
the `data` variable::

 import BaseIterativeData_t.data as mysubtree

The pattern is a :term:`CGNS/Python` list and
thus it should be copied before any modification:: 

 import BaseIterativeData_t
 import copy

 t=BaseIterativeData_t.data

 t1=copy.deepcopy(t)
 t2=copy.deepcopy(t)

For example, you can use *PAT.cgnslib* to create a `BaseIterativeData_t` node
with::

  data=C.newBaseIterativeData(None)

This call create the unique `BaseIterativeData_t` node (or sub-tree which is
the same in this case because we have only one node). The new node is returned,
the `None` argument means we do not define a parent node, it is up to the user
to add this new node in a existing children list.

Now we can use the *PAT.SIDS.BaseIterativeData_t* which creates the same
`BaseIterativeData_t` node as before, but also create the whole *SIDS*
sub-tree with default values, here is a snippet of this pattern::

  import CGNS.PAT.cgnslib      as C
  import CGNS.PAT.cgnskeywords as K

  data=C.newBaseIterativeData(None)
  C.newDataArray(data,K.NumberOfZones_s)
  C.newDataArray(data,K.NumberOfFamilies_s)
  C.newDataArray(data,K.ZonePointers_s)
  C.newDataArray(data,K.FamilyPointers_s)
  C.newDataArray(data,'{DataArray}')
  C.newDataClass(data)
  C.newDimensionalUnits(data)
  C.newUserDefinedData(data,'{UserDefinedData}')
  C.newDescriptor(data,'{Descriptor}')

You see all the mandatory and optional *SIDS* nodes are created, the user has
to set his own values in the resulting sub-tree using the *PAT.cgnslib* or the
*PAT.cgnsutils* functions.

.. _pat_cgnskeywords:

CGNS Keywords
-------------
Instead of generating a new doc from a file, the file itself is included here.
The purpose of `cgnskeywords.py` is to declare all constants as Python 
variables. This leads to several advantages:

 * You cannot make a typo on a name. For example, if you use 
   "ZoneGridConnectivity" as a plain string you may mistype it as
   "Zonegridconnectivity" or "ZoneGridConectivity" and this may silently
   produce a bad CGNS tree.

 * You can handle enumerate as lists. For example you have lists for 
   units: MassUnits_l, LengthUnits_l, AllDimensionalUnits_l, AllUnits_l

 * You can identify what is a CGNS reserved or recommended name or not.

.. literalinclude:: CGNS/PAT/cgnskeywords.py
   
.. _pat_cgnstypes:

CGNS Types
----------
.. USE THIS COMMAND FOR FILE UPDATE
.. python -c 'import CGNS.APP.sids.checktypes as C;C.generateSphinx("PATternMaker/cgnstypes.txt")'

.. toctree::
   
   cgnstypes

.. _pat_cgnserrors:

Error codes and functions
-------------------------

The errors are managed using exceptions. The base class is `cgnsException`, the
derived classes are in the list below, for each class you can have several
error codes. For example you can catch `cgnsNameError` and have a more
detailled error diagnostic with the error code::

  try:
    CGU.checkName('.')
  except CGE.cgnsNameError:
    # skip exception
    # a cgnsNameError is a cgnsException
    pass

  try:
    CGU.checkName('zapzap/s')
  except CGE.cgnsException,why:
    # get message and print it
    # actually 'why' is the exception object but print calls its __str__ 
    print why

  try:
    CGU.checkName('')
  except CGE.cgnsNameError,exc:
    # a cgnsException has a 'code' attribute (the integer error code)
    # a 'value' attribute with a tuple of arguments set at raise time
    # a cgnsNameError is a cgnsException
    if (exc.code==21): print 'Cannot find node ',exc.value

.. _cgnsnameerror:

cgnsNameError
~~~~~~~~~~~~~

+-------+--------------------------------------------------------------------+
| code  | Message                                                            |
+=======+====================================================================+
| 21    | No node with name [%s]                                             |
+-------+--------------------------------------------------------------------+
| 22    | Node name should have type string                                  |
+-------+--------------------------------------------------------------------+
| 23    | Empty string is not allowed for a node name                        |
+-------+--------------------------------------------------------------------+
| 24    | Node name should not contain a '/'                                 |
+-------+--------------------------------------------------------------------+
| 25    | Node name length should not be greater than 32 chars               |
+-------+--------------------------------------------------------------------+
| 102   | Duplicated child name [%s] in [%s]                                 |
+-------+--------------------------------------------------------------------+

.. _cgnsnodeerror:

cgnsNodeError
~~~~~~~~~~~~~

+-------+--------------------------------------------------------------------+
| code  | Message                                                            |
+=======+====================================================================+
| 1     | Node is empty !                                                    |
+-------+--------------------------------------------------------------------+
| 2     | Node should be a list of <name, value, children, type>             |
+-------+--------------------------------------------------------------------+
| 3     | Node name should be a string                                       |
+-------+--------------------------------------------------------------------+
| 4     | Node [%s] children list should be a list                           |
+-------+--------------------------------------------------------------------+
| 5     | Node [%s] bad value: should be a numpy object                      |
+-------+--------------------------------------------------------------------+

.. _cgnstypeerror:

cgnsTypeError
~~~~~~~~~~~~~

+-------+--------------------------------------------------------------------+
| code  | Message                                                            |
+=======+====================================================================+
| 103   | Node type of [%s] not [%s]                                         |
+-------+--------------------------------------------------------------------+
| 104   | Node type of [%s] not in %s                                        |
+-------+--------------------------------------------------------------------+

.. _cgnsvalueerror:

cgnsValueError
~~~~~~~~~~~~~~
+-------+--------------------------------------------------------------------+
| code  | Message                                                            |
+=======+====================================================================+
| 000   |                                                                    |
+-------+--------------------------------------------------------------------+

-----

.. include:: ../doc/Intro/glossary.txt

.. _pat_index:

PAT Index
---------

* :ref:`genindex`

.. -------------------------------------------------------------------------
