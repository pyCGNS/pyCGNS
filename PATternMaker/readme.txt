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
| Add child                                                   | :py:func:`addChild <CGNS.PAT.cgnsutils.addChild>`                                       |                 
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of children names                                      | :py:func:`childrenNames <CGNS.PAT.cgnsutils.childrenNames>`                             |                 
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Delete child                                                | :py:func:`removeChildByName <CGNS.PAT.cgnsutils.removeChildByName>`                     |                 
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Node structure and contents**                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks basic node structure                                 | :py:func:`checkNode <CGNS.PAT.cgnsutils.checkNode>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node is a CGNS/Python tree root         | :py:func:`checkRootNode <CGNS.PAT.cgnsutils.checkRootNode>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node is CGNS/Python compliant           | :py:func:`checkNodeCompliant <CGNS.PAT.cgnsutils.checkNodeCompliant>`                   |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the node is of arg CGNS/SIDS type           | :py:func:`checkNodeType <CGNS.PAT.cgnsutils.checkNodeType>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Returns True if the parent is of arg CGNS/SIDS type         | :py:func:`checkParentType <CGNS.PAT.cgnsutils.checkParentType>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the node name has CGNS/SIDS correct syntax        | :py:func:`checkNodeName <CGNS.PAT.cgnsutils.checkNodeName>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the name has CGNS/SIDS correct syntax             | :py:func:`checkName <CGNS.PAT.cgnsutils.checkName>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if two nodes have same contents                        | :py:func:`checkSameNode <CGNS.PAT.cgnsutils.checkSameNode>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if arg name is not already in the parent children list | :py:func:`checkDuplicatedName <CGNS.PAT.cgnsutils.checkDuplicatedName>`                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| True if arg name is the parent children list                | :py:func:`checkHasChildName <CGNS.PAT.cgnsutils.checkHasChildName>`                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if arg path is CGNS/SIDS compliant                   | :py:func:`checkPath <CGNS.PAT.cgnsutils.checkPath>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the value is a numpy.ndarray                      | :py:func:`checkArray <CGNS.PAT.cgnsutils.checkArray>`                                   |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the value is a numpy.ndarray of type C1           | :py:func:`checkArrayChar <CGNS.PAT.cgnsutils.checkArrayChar>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the value is a numpy.ndarray of type R4 or R8     | :py:func:`checkArrayReal <CGNS.PAT.cgnsutils.checkArrayReal>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Checks if the value is a numpy.ndarray of type I4 or I8     | :py:func:`checkArrayInteger <CGNS.PAT.cgnsutils.checkArrayInteger>`                     |
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
| True if first path is prefix of second                      | :py:func:`hasSameRootPath <CGNS.PAT.cgnsutils.hasSameRootPath>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+ 
|                                                             | **Search and Retrieve**                                                                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return  a node from its path                                | :py:func:`getNodeByPath <CGNS.PAT.cgnsutils.getNodeByPath>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return  a node value from node path                         | :py:func:`getValueByPath <CGNS.PAT.cgnsutils.getValueByPath>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return  the parent node of the arg node                     | :py:func:`getParentFromNode <CGNS.PAT.cgnsutils.getParentFromNode>`                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return  list of node children from path                     | :py:func:`getChildrenByPath <CGNS.PAT.cgnsutils.getChildrenByPath>`                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return  a node CGNS/SIDS type from path                     | :py:func:`getTypeByPath <CGNS.PAT.cgnsutils.getTypeByPath>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| All nodes matching the unordered set of CGNS/SIDS types     | :py:func:`getAllNodesByTypeSet <CGNS.PAT.cgnsutils.getAllNodesByTypeSet>`               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| All nodes matching the ordered list of CGNS/SIDS types      | :py:func:`getAllNodesByTypeList <CGNS.PAT.cgnsutils.getAllNodesByTypeList>`             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| All nodes matching a list of names and CGNS/SIDS types      | :py:func:`getAllNodesByTypeOrNameList <CGNS.PAT.cgnsutils.getAllNodesByTypeOrNameList>` |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Iterator, get next child by type/name order                 | :py:func:`getNextChildSortByType <CGNS.PAT.cgnsutils.getNextChildSortByType>`           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+             
|                                                             | **CGNS/SIDS information**                                                               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of allowed CGNS/SIDS types as child of node            | :py:func:`getNodeAllowedChildrenTypes <CGNS.PAT.cgnsutils.getNodeAllowedChildrenTypes>` |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of allowed CGNS/SIDS data types for this node          | :py:func:`getNodeAllowedDataTypes <CGNS.PAT.cgnsutils.getNodeAllowedDataTypes>`         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| List of all allowed CGNS/SIDS parent types paths            | :py:func:`getAllParentTypePaths <CGNS.PAT.cgnsutils.getAllParentTypePaths>`             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Value manipulation**                                                                  |               
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return node value shape (numpy.ndarray)                     | :py:func:`getValueShape <CGNS.PAT.cgnsutils.getValueShape>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return node value data type as CGNS/SIDS data type          | :py:func:`getValueDataType <CGNS.PAT.cgnsutils.getValueDataType>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return value data type as CGNS/SIDS data type               | :py:func:`getValueType <CGNS.PAT.cgnsutils.getValueType>`                               |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return True if the node has a value                         | :py:func:`hasValue <CGNS.PAT.cgnsutils.hasValue>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return True if the node has a numpy.ndarray fortran order   | :py:func:`hasFortranFlag <CGNS.PAT.cgnsutils.hasFortranFlag>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Return True if the node value data type is correct          | :py:func:`hasValueDataType <CGNS.PAT.cgnsutils.hasValueDataType>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Create a numpy.ndarray from a string                        | :py:func:`setStringAsArray <CGNS.PAT.cgnsutils.setStringAsArray>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Create a numpy.ndarray from list of strings                 | :py:func:`concatenateForCharArray2D <CGNS.PAT.cgnsutils.concatenateForCharArray2D>`     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Create a numpy.ndarray from list of list of strings         | :py:func:`concatenateForCharArray3D <CGNS.PAT.cgnsutils.concatenateForCharArray3D>`     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the node value                                          | :py:func:`getValue <CGNS.PAT.cgnsutils.getValue>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Set the node value                                          | :py:func:`setValue <CGNS.PAT.cgnsutils.setValue>`                                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Check node value and arg string                             | :py:func:`stringValueMatches <CGNS.PAT.cgnsutils.stringValueMatches>`                   |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Copy numpy.ndarray with flags                               | :py:func:`copyArray <CGNS.PAT.cgnsutils.copyArray>`                                     |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             |                                                                                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                             | **Path functions**                                                                      |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the path of a node not including root name              | :py:func:`getPathFromNode <CGNS.PAT.cgnsutils.getPathFromNode>`                         |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get the path of a node including root name                  | :py:func:`getPathFromRoot <CGNS.PAT.cgnsutils.getPathFromRoot>`                         |
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
| Get the first item of the path                              | :py:func:`hasFirstPathItem <CGNS.PAT.cgnsutils.hasFirstPathItem>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Remove the first item of the path                           | :py:func:`removeFirstPathItem <CGNS.PAT.cgnsutils.removeFirstPathItem>`                 |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get path with CGNS/Python root removed                      | :py:func:`getPathNoRoot <CGNS.PAT.cgnsutils.getPathNoRoot>`                             |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Get list of types of nodes along the arg path               | :py:func:`getPathAsTypes <CGNS.PAT.cgnsutils.getPathAsTypes>`                           |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Normalizes the arg path                                     | :py:func:`getPathNormalize <CGNS.PAT.cgnsutils.getPathNormalize>`                       |
+-------------------------------------------------------------+-----------------------------------------------------------------------------------------+
| Find the common ancestor of path list                       | :py:func:`getPathListCommonAncestor <CGNS.PAT.cgnsutils.getPathListCommonAncestor>`     |
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

+-----------+----------------------------------------------------------------------------------------+
|   SIDS    | Function                                                                               |
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newCGNSTree  <CGNS.PAT.cgnslib.newCGNSTree>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|   6.2     | :py:func:`newCGNSBase  <CGNS.PAT.cgnslib.newCGNSBase>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|   4.1     | :py:func:`newDataClass  <CGNS.PAT.cgnslib.newDataClass>`                               | 
+-----------+----------------------------------------------------------------------------------------+
|   4.2     | :py:func:`newDescriptor  <CGNS.PAT.cgnslib.newDescriptor>`                             | 
+-----------+----------------------------------------------------------------------------------------+
|   4.3     | :py:func:`newDimensionalUnits  <CGNS.PAT.cgnslib.newDimensionalUnits>`                 | 
+-----------+----------------------------------------------------------------------------------------+
|   4.4     | :py:func:`newDimensionalExponents  <CGNS.PAT.cgnslib.newDimensionalExponents>`         | 
+-----------+----------------------------------------------------------------------------------------+
|   4.5     | :py:func:`newGridLocation  <CGNS.PAT.cgnslib.newGridLocation>`                         | 
+-----------+----------------------------------------------------------------------------------------+
|   4.6     | :py:func:`newIndexArray  <CGNS.PAT.cgnslib.newIndexArray>`                             | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newPointList  <CGNS.PAT.cgnslib.newPointList>`                               | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newPointRange  <CGNS.PAT.cgnslib.newPointRange>`                             | 
+-----------+----------------------------------------------------------------------------------------+
|   4.8     | :py:func:`newRind  <CGNS.PAT.cgnslib.newRind>`                                         | 
+-----------+----------------------------------------------------------------------------------------+
|   5.1.1   | :py:func:`newDataConversion  <CGNS.PAT.cgnslib.newDataConversion>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newSimulationType  <CGNS.PAT.cgnslib.newSimulationType>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newBase  <CGNS.PAT.cgnslib.newBase>`                                         | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newOrdinal  <CGNS.PAT.cgnslib.newOrdinal>`                                   | 
+-----------+----------------------------------------------------------------------------------------+
|   6.3     | :py:func:`newZone  <CGNS.PAT.cgnslib.newZone>`                                         | 
+-----------+----------------------------------------------------------------------------------------+
|   7.1     | :py:func:`newGridCoordinates  <CGNS.PAT.cgnslib.newGridCoordinates>`                   | 
+-----------+----------------------------------------------------------------------------------------+
|   5.1     | :py:func:`newDataArray  <CGNS.PAT.cgnslib.newDataArray>`                               | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newDiscreteData  <CGNS.PAT.cgnslib.newDiscreteData>`                         | 
+-----------+----------------------------------------------------------------------------------------+
|   7.3     | :py:func:`newElements  <CGNS.PAT.cgnslib.newElements>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|   9.2     | :py:func:`newZoneBC  <CGNS.PAT.cgnslib.newZoneBC>`                                     | 
+-----------+----------------------------------------------------------------------------------------+
|   9.3     | :py:func:`newBC  <CGNS.PAT.cgnslib.newBC>`                                             | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newBoundary  <CGNS.PAT.cgnslib.newBoundary>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|   9.4     | :py:func:`newBCDataSet  <CGNS.PAT.cgnslib.newBCDataSet>`                               | 
+-----------+----------------------------------------------------------------------------------------+
|   9.5     | :py:func:`newBCData  <CGNS.PAT.cgnslib.newBCData>`                                     | 
+-----------+----------------------------------------------------------------------------------------+
|   9.6     | :py:func:`newBCProperty  <CGNS.PAT.cgnslib.newBCProperty>`                             | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newCoordinates  <CGNS.PAT.cgnslib.newCoordinates>`                           | 
+-----------+----------------------------------------------------------------------------------------+
|   7.5     | :py:func:`newAxisymmetry  <CGNS.PAT.cgnslib.newAxisymmetry>`                           | 
+-----------+----------------------------------------------------------------------------------------+
|   7.6     | :py:func:`newRotatingCoordinates  <CGNS.PAT.cgnslib.newRotatingCoordinates>`           | 
+-----------+----------------------------------------------------------------------------------------+
|   7.7     | :py:func:`newFlowSolution  <CGNS.PAT.cgnslib.newFlowSolution>`                         | 
+-----------+----------------------------------------------------------------------------------------+
|   8.1     | :py:func:`newZoneGridConnectivity  <CGNS.PAT.cgnslib.newZoneGridConnectivity>`         | 
+-----------+----------------------------------------------------------------------------------------+
|   8.2     | :py:func:`newGridConnectivity1to1  <CGNS.PAT.cgnslib.newGridConnectivity1to1>`         | 
+-----------+----------------------------------------------------------------------------------------+
|   8.4     | :py:func:`newGridConnectivity  <CGNS.PAT.cgnslib.newGridConnectivity>`                 | 
+-----------+----------------------------------------------------------------------------------------+
|   8.6     | :py:func:`newGridConnectivityProperty  <CGNS.PAT.cgnslib.newGridConnectivityProperty>` | 
+-----------+----------------------------------------------------------------------------------------+
|   8.6.2   | :py:func:`newAverageInterface  <CGNS.PAT.cgnslib.newAverageInterface>`                 | 
+-----------+----------------------------------------------------------------------------------------+
|   8.7     | :py:func:`newOversetHoles  <CGNS.PAT.cgnslib.newOversetHoles>`                         | 
+-----------+----------------------------------------------------------------------------------------+
|  10.1     | :py:func:`newFlowEquationSet  <CGNS.PAT.cgnslib.newFlowEquationSet>`                   | 
+-----------+----------------------------------------------------------------------------------------+
|  10.2     | :py:func:`newGoverningEquations  <CGNS.PAT.cgnslib.newGoverningEquations>`             | 
+-----------+----------------------------------------------------------------------------------------+
|  10.4     | :py:func:`newGasModel  <CGNS.PAT.cgnslib.newGasModel>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|  10.6     | :py:func:`newThermalConductivityModel  <CGNS.PAT.cgnslib.newThermalConductivityModel>` | 
+-----------+----------------------------------------------------------------------------------------+
|  10.5     | :py:func:`newViscosityModel  <CGNS.PAT.cgnslib.newViscosityModel>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|  10.7.1   | :py:func:`newTurbulenceClosure  <CGNS.PAT.cgnslib.newTurbulenceClosure>`               | 
+-----------+----------------------------------------------------------------------------------------+
|  10.7.2   | :py:func:`newTurbulenceModel  <CGNS.PAT.cgnslib.newTurbulenceModel>`                   | 
+-----------+----------------------------------------------------------------------------------------+
|  10.8     | :py:func:`newThermalRelaxationModel  <CGNS.PAT.cgnslib.newThermalRelaxationModel>`     | 
+-----------+----------------------------------------------------------------------------------------+
|  10.9     | :py:func:`newChemicalKineticsModel  <CGNS.PAT.cgnslib.newChemicalKineticsModel>`       | 
+-----------+----------------------------------------------------------------------------------------+
|  10.10.1  | :py:func:`newEMElectricFieldModel  <CGNS.PAT.cgnslib.newEMElectricFieldModel>`         | 
+-----------+----------------------------------------------------------------------------------------+
|  10.10.2  | :py:func:`newEMMagneticFieldModel  <CGNS.PAT.cgnslib.newEMMagneticFieldModel>`         | 
+-----------+----------------------------------------------------------------------------------------+
|  10.10.3  | :py:func:`newEMConductivityModel  <CGNS.PAT.cgnslib.newEMConductivityModel>`           | 
+-----------+----------------------------------------------------------------------------------------+
|  11.1.1   | :py:func:`newBaseIterativeData  <CGNS.PAT.cgnslib.newBaseIterativeData>`               | 
+-----------+----------------------------------------------------------------------------------------+
|  11.1.2   | :py:func:`newZoneIterativeData  <CGNS.PAT.cgnslib.newZoneIterativeData>`               | 
+-----------+----------------------------------------------------------------------------------------+
|  11.2     | :py:func:`newRigidGridMotion  <CGNS.PAT.cgnslib.newRigidGridMotion>`                   | 
+-----------+----------------------------------------------------------------------------------------+
|  12.1     | :py:func:`newReferenceState  <CGNS.PAT.cgnslib.newReferenceState>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|  12.3     | :py:func:`newConvergenceHistory  <CGNS.PAT.cgnslib.newConvergenceHistory>`             | 
+-----------+----------------------------------------------------------------------------------------+
|  12.5     | :py:func:`newIntegralData  <CGNS.PAT.cgnslib.newIntegralData>`                         | 
+-----------+----------------------------------------------------------------------------------------+
|  12.6     | :py:func:`newFamily  <CGNS.PAT.cgnslib.newFamily>`                                     | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newFamilyName  <CGNS.PAT.cgnslib.newFamilyName>`                             | 
+-----------+----------------------------------------------------------------------------------------+
|  12.7     | :py:func:`newGeometryReference  <CGNS.PAT.cgnslib.newGeometryReference>`               | 
+-----------+----------------------------------------------------------------------------------------+
|  12.8     | :py:func:`newFamilyBC  <CGNS.PAT.cgnslib.newFamilyBC>`                                 | 
+-----------+----------------------------------------------------------------------------------------+
|  11.3     | :py:func:`newArbitraryGridMotion  <CGNS.PAT.cgnslib.newArbitraryGridMotion>`           | 
+-----------+----------------------------------------------------------------------------------------+
|  12.10    | :py:func:`newUserDefinedData  <CGNS.PAT.cgnslib.newUserDefinedData>`                   | 
+-----------+----------------------------------------------------------------------------------------+
|  12.11    | :py:func:`newGravity  <CGNS.PAT.cgnslib.newGravity>`                                   | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newField  <CGNS.PAT.cgnslib.newField>`                                       | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newModel  <CGNS.PAT.cgnslib.newModel>`                                       | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newDiffusionModel  <CGNS.PAT.cgnslib.newDiffusionModel>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newParentElements  <CGNS.PAT.cgnslib.newParentElements>`                     | 
+-----------+----------------------------------------------------------------------------------------+
|           | :py:func:`newParentElementsPosition  <CGNS.PAT.cgnslib.newParentElementsPosition>`     | 
+-----------+----------------------------------------------------------------------------------------+


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
