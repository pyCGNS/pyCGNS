.. -------------------------------------------------------------------------
.. pyCGNS.PAT - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.PAT
========

The *PATtern* module provides the user with functions dedicated to
:ref:`CGNS/Python <mapix:reference_sids_to_python>` trees.
The :ref:`PAT.cgnslib <pat_cgnslib>` module uses the *SIDS* compliant data 
structures, you
can create, read, check, modify some 
:ref:`CGNS/Python <mapix:reference_sids_to_python>` sub-trees related to a
*SIDS* type.
The :ref:`PAT.cgnsutils <pat_cgnsutils>` provides utility fonctions 
for raw :ref:`CGNS/Python <mapix:reference_sids_to_python>` trees or nodes.
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
the :ref:`CGNS/Python <mapix:reference_sids_to_python>` nodes,
sub-trees or trees as arguments, you can manipulate tree paths, links,
values.  Functions are not gathered into a class because we want them
to proceed on standard 
:ref:`CGNS/Python <mapix:reference_sids_to_python>` trees. Most functions have an
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

 * Node life cycle: :py:func:`nodeCreate <CGNS.PAT.cgnsutils.nodeCreate>` -
   :py:func:`nodeCopy <CGNS.PAT.cgnsutils.nodeCopy>` -
   :py:func:`nodeDelete <CGNS.PAT.cgnsutils.nodeDelete>` -
   :py:func:`nodeLink <CGNS.PAT.cgnsutils.nodeLink>`

 * Check functions: :py:func:`checkNode <CGNS.PAT.cgnsutils.checkNode>` -
   :py:func:`checkRootNode <CGNS.PAT.cgnsutils.checkRootNode>` -
   :py:func:`checkNodeType <CGNS.PAT.cgnsutils.checkNodeType>` -
   :py:func:`checkNodeName <CGNS.PAT.cgnsutils.checkNodeName>` -
   :py:func:`checkSameNode <CGNS.PAT.cgnsutils.checkSameNode>` -
   :py:func:`checkDuplicatedName <CGNS.PAT.cgnsutils.checkDuplicatedName>` -
   :py:func:`checkPath <CGNS.PAT.cgnsutils.checkPath>` -
   :py:func:`checkLink <CGNS.PAT.cgnsutils.checkLink>`-

 * Node true/false tests: :py:func:`hasChildType <CGNS.PAT.cgnsutils.hasChildType>` -
   :py:func:`hasAncestorType <CGNS.PAT.cgnsutils.hasAncestorType>` -
   :py:func:`hasChildName <CGNS.PAT.cgnsutils.hasChildName>` -
   :py:func:`hasAncestorName <CGNS.PAT.cgnsutils.hasAncestorName>` -
   :py:func:`hasValue <CGNS.PAT.cgnsutils.hasValue>` -
   :py:func:`hasValueDataType <CGNS.PAT.cgnsutils.hasValueDataType>` -
   :py:func:`hasChildLink <CGNS.PAT.cgnsutils.hasChildLink>` -
   :py:func:`hasAncestorLink <CGNS.PAT.cgnsutils.hasAncestorLink>` -
   :py:func:`hasValueFlags <CGNS.PAT.cgnsutils.hasValueFlags>` -	

 * Data retrieval simple functions: :py:func:`getNodeByPath <CGNS.PAT.cgnsutils.getNodeByPath>` -
   :py:func:`getValueByPath <CGNS.PAT.cgnsutils.getValueByPath>` -
   :py:func:`getChildrenByPath <CGNS.PAT.cgnsutils.getChildrenByPath>` -
   :py:func:`getTypeByPath <CGNS.PAT.cgnsutils.getTypeByPath>`

 * Data retrieval specialized functions: :py:func:`getAllNodesByTypeSet <CGNS.PAT.cgnsutils.getAllNodesByTypeSet>` -
   :py:func:`getNodeAllowedChildrenTypes <CGNS.PAT.cgnsutils.getNodeAllowedChildrenTypes>` -
   :py:func:`getNodeAllowedDataTypes <CGNS.PAT.cgnsutils.getNodeAllowedDataTypes>` -

 * Node value manipulation: :py:func:`getValueShape <CGNS.PAT.cgnsutils.getValueShape>` -
   :py:func:`getValueDataType <CGNS.PAT.cgnsutils.getValueDataType>` -
   :py:func:`hasValue <CGNS.PAT.cgnsutils.hasValue>` -
   :py:func:`hasValueDataType <CGNS.PAT.cgnsutils.hasValueDataType>` -
   :py:func:`getValueByPath <CGNS.PAT.cgnsutils.getValueByPath>` -


 * Path retrieval functions: :py:func:`getPathFromNode <CGNS.PAT.cgnsutils.getPathFromNode>` -
   :py:func:`getPathFullTree <CGNS.PAT.cgnsutils.getPathFullTree>` -
   :py:func:`getPathByNameFilter <CGNS.PAT.cgnsutils.getPathByNameFilter>` -
   :py:func:`getPathByTypeFilter <CGNS.PAT.cgnsutils.getPathByTypeFilter>` -

 * Path manipulation: :py:func:`getPathToList <CGNS.PAT.cgnsutils.getPathToList>` -
   :py:func:`getPathAncestor <CGNS.PAT.cgnsutils.getPathAncestor>` -
   :py:func:`getPathLeaf <CGNS.PAT.cgnsutils.getPathLeaf>` -
   :py:func:`getPathNoRoot <CGNS.PAT.cgnsutils.getPathNoRoot>` -
   :py:func:`getPathAsTypes <CGNS.PAT.cgnsutils.getPathAsTypes>` -
   :py:func:`getPathNormalize <CGNS.PAT.cgnsutils.getPathNormalize>` -

 * Link manipulation:

.. note:

   The functions are not gathered into one or more classes because we want these to
   



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

The pattern is a :ref:`CGNS/Python <mapix:reference_sids_to_python>` list and
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
