.. -------------------------------------------------------------------------
.. pyCGNS.MAP - CFD General Notation System - SIDS-to-Python MAPping           
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

MAP
===

Quick start
-----------
The MAPper is a module implementing the SIDS-to-Python CGNS mapping.
The MAP module loads and saves CGNS/HDF5 files as Python trees.

A simple exemple to load a *CGNS/HDF5* file as a *CGNS/Python* tree::

  import CGNS.MAP

  (tree,links)=CGNS.MAP.load("./T0.cgns",CGNS.MAP.S2P_FOLLOWLINKS)

The ``tree`` value contains the actual CGNS/Python tree with linked-to
files included (because the ``S2P_FOLLOWLINKS`` flag is *on*) and the
``links`` value is a list of links found during the *HDF5* file parse.

User interface
--------------
MAP is a lightweight module, its purpose is to be as small as possible
in order to be embedded separatly in an application 
(see :ref:`Embbeded MAP <reference_embedded_map>` .

Functions
~~~~~~~~~
There are two functions: the ``load`` and the ``save``. The ``load`` reads
a CGNS/HDF5 file and produces a CGNS/Python tree. The ``save`` takes a 
CGNS/Python tree and writes the contents in a CGNS/HDF5 file::

 (tree,links)=CGNS.MAP.load(filename,flags,threshold,depth,path)

 status=CGNS.MAP.save(filename,tree,links,flags,threshold,depth,path)

The arguments and the return values are:

 * tree
   The ``tree`` is the list representing the CGNS/Python tree. 
   The structure of a ``tree`` list is detailled 
   in :ref:`SIDS-to-Python <reference_sids_to_python>`.
   There is no link information in this tree either for *load* or for *save*. 

   During the *load*, the links are silently replaced by the linked-to tree 
   they are referring. The ``links`` value keeps track of these link 
   references found while parsing the CGNS/HDF5 file. 

   During the *save*, the tree is splitted into separate files/nodes depending
   on the references found in the ``links`` value.

 * links
   The ``links`` is a list with the link node information. It is returned
   by a *load* and used as command parameters during the *save*. You can write
   your own ``links`` list or change the list you obtain after a *load*.
   The structure of a ``links`` list is detailled 
   in :ref:`SIDS-to-Python <reference_sids_to_python>`.

Flags
~~~~~
The flags are integers that can be OR-ed or XOR-ed to set/unset specific
behavior during the load and the save.
The boolean operators are used for the flag settings::

 flags=CGNS.MAP.S2P_FOLLOWLINKS|CGNS.MAP.S2P_TRACE

 flags =flags&~CGNS.MAP.S2P_TRACE
 flags&=~CGNS.MAP.S2P_TRACE  

The table below gives the `CGNS.MAP` flags.

 +-----------------------+------------------------------------------------------+
 | *Flag variable*       | *Function*                                           |
 +=======================+======================================================+
 | ``S2P_NONE``          | Clear all flags, set to zero.                        |
 +-----------------------+------------------------------------------------------+
 | ``S2P_ALL``           | Set all flags, set to one.                           |
 +-----------------------+------------------------------------------------------+
 | ``S2P_TRACE``         | Set the trace on, messages are sent to 'stdout'      |
 +-----------------------+------------------------------------------------------+
 | ``S2P_FOLLOWLINKS``   | Continue to parse the linked-to tree \(1)            |
 +-----------------------+------------------------------------------------------+
 | ``S2P_MERGELINKS``    | Forget all link specifications.  \(2)                |
 +-----------------------+------------------------------------------------------+
 | ``S2P_COMPRESS``      | Sets the compress flag for 'DataArray_t' \(2)        |
 +-----------------------+------------------------------------------------------+
 | ``S2P_NOTRANSPOSE``   | Do not transpose *dimensions* during load and save.  |
 +-----------------------+------------------------------------------------------+
 | ``S2P_NOOWNDATA``     | Forces the `numpy` flag ``\~NPY_OWNDATA`` \(1) \(3)  |
 +-----------------------+------------------------------------------------------+
 | ``S2P_NODATA``        | Do not load large 'DataArray_t' \(2) \(4)            |
 +-----------------------+------------------------------------------------------+
 | ``S2P_UPDATE``        | not used                                             |
 +-----------------------+------------------------------------------------------+
 | ``S2P_DELETEMISSING`` | not used                                             |
 +-----------------------+------------------------------------------------------+

There is no requirements or check on which flag can or cannot be associated
with another flag.

**Remarks:**
  
  (1) Only when you are *loading* a tree. 

  (2) Only when you are *saving* a tree.

  (3) Which means all ``DataArray_t` actual memory zones will **NOT** be
      released by Python.

  (4) The term `large` has to be defined. The *save* will **NOT** check if
      the CGNS/Python tree was performed with the ``S2P_NODATA`` flag on,
      then you have to check by yourself that your *save* will not overwrite
      an existing file with empty data!

Examples
--------


.. _reference_sids_to_python:

CGNS/Python mapping 
===================

The CGNS/Python mapping defines a **tree** structure composed of a basic
**node** and a **links** structure.

CGNS/Python Tree
~~~~~~~~~~~~~~~~
An CGNS/Python node is mapped as follow:

 +---------+---------------------------+ 
 |*Name*   | *string*                  |
 +=========+===========================+ 
 |Value    | *numpy* array             |
 +---------+---------------------------+ 
 |Children | list of CGNS/Python nodes |
 +---------+---------------------------+ 
 |Type     | string                    |
 +---------+---------------------------+ 

The node structure is a python sequence (i.e. list or tuple), composed
of four entries: the name, the value, the list of children and the
type.

  The *name* is a Python string, it should not be empty. The name should
  not have more than 32 chars and should not have ``/`` in it. The names
  ``.`` (a single dot) And ``..`` (dot dot) are forbidden (but names with dot
  and something else are allowed, for example ``Zone.001`` is ok).

  The representation of *values* uses the numpy library array. It makes
  it possible to share an actual memory zone (C or Fortran array) with
  the Python object. The numpy mapping of the values is detailled
  hereafter. An empty value should be represented as None, any other
  value is forbidden.

  The *children list* can be a list or a tuple. The use of a list is
  strongly recommended but not mandatory. A read-only tree can be
  declared as a tuple. It is the responsibility of the applications to
  parse sequences wether these are lists or tuples. A node without child
  has the empty list ``[]`` as children list.

  The *type* is the Python string representation of the CGNS/SIDS type
  (i.e. it is the same for CGNS/ADF or CGNS/HDF5). A type string cannot be
  empty.  

It is possible to declare a CGNS/Python node as a textual
representation. There is a exemple of a zone connectivity sub-tree
with the CGNS/Python in textual mode, a simple ``PointRange`` node with
two 3D indices::
 
 pointrange=['PointRange',
             numpy.array([[1,25],[1,9],[1,1]],dtype=numpy.int32,order='Fortran'),
             [],
             'IndexRange_t']

The evaluation by the Python interpreter creates a CGNS/Python
compliant node as a Python list. Please note the types of this
pointrange node, there is only native Python types (list, string,
integer) and numpy types or enumerates.  numpy array mapping

A CGNS/Python node value is a *numpy* array, this python object contains
the number of dimensions, the dimensions, the data type and the actual
data array. The *numpy* end-user interface makes it possible to define
some of these required data as deduction of required
parameters. The number of dimensions is the size of the so-called
shape. The dimensions can be forced for empty values or can be deduced
from the data itself::

  a=numpy.array([1.4])
  b=numpy.ones((5,7,3),'i')

The first declaration has dimension 1, number of dims 1, data type
``float64``, all deduced from the data declaration, the second has
dimensions ``(5,3,7)``, number of dimensions 3, data type set as
``int32``. 

The 'Fortran' order should be used for the array data indexing. The
shape of an array gives the Fortran-indexing dimensions. All
CGNS/Python compliant applications should provide a 'Fortran' order
indexing at the end-user interface. For example, the ``PointRange`` 
above **should** have a *shape* of ``(3,2)`` and **not** ``(2,3)``.
A creation using a list of point would lead to a wrong array, because
Python and *numpy* are using C-order convention by default. At this level
(end-user level in the Python interpreter), the *Fortran* flag has no effect.
The C-order::

 >>> a=numpy.array([[1,2,3],[4,5,6]],dtype=numpy.int32)
 >>> numpy.isfortran(a)
 False
 >>> a[0]
 [1,2,3]
 >>> a.shape
 (2,3)

The Fortran-order::

 >>> a=numpy.array([[1,4],[2,5],[3,6]],dtype=numpy.int32)
 >>> numpy.isfortran(a)
 False
 >>> a[0]
 [1,4]
 >>> a.shape
 (3,2)

The fortran flag is required for internal purpose, in particular **CGNS.MAP**
fails if you try to save a CGNS/Python tree without this numpy flag on.
The correct creation of the array above is then::

 >>> a=numpy.array([[1,4],[2,5],[3,6]],dtype=numpy.int32,order='Fortran')
 >>> numpy.isfortran(a)
 True
 >>> a[0]
 [1,4]
 >>> a.shape
 (3,2)

There is another example switching from one order to another, this is used
to add a point in a list in an easier way::

 >>> a=numpy.array([[1,4],[2,5],[3,6]],dtype=numpy.int32,order='Fortran')
 >>> a
 array([[1, 4],
        [2, 5],
        [3, 6]], dtype=int32)
  
 >>> a=numpy.array(a.T.tolist()+[[7,8,9]],dtype=numpy.int32,order='Fortran')
 >>> a
 array([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]], dtype=int32)

The type of the data you use at the creation time is very important,
the `numpy` type is associated to the `ADF` type required by the
CGNS/SIDS. A bad data type, even if it silently looks like the result
you want, would raise an error.
The required mapping for the end-user interface uses the types :

+-------------+-----------------+-----------+
|*ADF type*   |*Numpy type(s)*	|*Remarks*  |
+=============+=================+===========+
|`I4`         |`'i' int32`    	| \(1)      |
+-------------+-----------------+-----------+
|`I8`         |`'l' int64`     	| \(2)      |
+-------------+-----------------+-----------+
|`R4`         |`'f' float32`   	| \(3)      |
+-------------+-----------------+-----------+
|`R8`         |`'d' float64`   	| \(4)      |
+-------------+-----------------+-----------+
|`C1`         |`'\|S1'`        	| \(5)      |
+-------------+-----------------+-----------+

All other `ADF` or `numpy` types are ignored. The string type is a bit special,
see the remark (5) about the strings used in `numpy` arrays.

**Remarks:**

 (1)  
    The 32bits precision has to be forced, the default
    integer size in python the ``int64`` data type.
    To create an `I4` array, you can use:: 
      numpy.array([1,2,3],'i',order='Fortran')

 (2) 
    The 64bits precision is the default integer in python.
    To create an `I8` array, you can use:: 
      numpy.array([1,2,3],order='Fortran')

 (3)  
    The 32bits precision has to be forced, the default 
    float size in python is ``float64``.
    To create an `R4` array, you can use:: 
      numpy.array([1.4],'f',order='Fortran')

 (4)  
    The 64bits precision is the default float in python.
    To create an `R8` array, you can use:: 
      numpy.array([1.4],order='Fortran')

 (5)  
    The array has to be created as a char multi-dimensionnal 
    array. An incorrect creation with a simple statement     
    such as: ``numpy.array('GoverningEquations')`` produces  
    a *wrong* zero dimension array. The correct creation for 
    a single value could be:                                 
    ``numpy.array(tuple('GoverningEquations'),'|S1')``      
    where the shape (i.e. the dimensions of the array)       
    is ``(18,)``. In the case of a fixed size                
    multi-dimensionnal string array, each entry should be    
    split as a sequence with a fixed max size (usually 32    
    chars)::                                                 
      numpy.array([
      tuple('%-32s'%'FlowSolution#001'),
      tuple('%-32s'%'FlowSolution#002')
      ],'|S32',order='Fortran').T
    The shape of the resulting array is ``(32,2)`` please note the ``T``
    at the end of the command, this produces the transpose.
    You can use a ``S32`` or a ``|S1`` type directive.
    An important point in this `string` as an array is the trailing
    `spaces` you have to fill the array cell. You have to use a 
    ``string.strip`` before any string operation unless your Python
    application is aware of this *forced* size.

CGNS/Python links
~~~~~~~~~~~~~~~~~
The **links** are used to set and get CGNS symbolic links information.
A CGNS/Python tree cannot have embedded links, as this tree is a list of
lists making a link to another list is non-sense in Python [#n1]_.
The **links** list is an unsorted list of *link-entries* with only one 
entry per link. A *link-entry* is an ordered list of Python string values:

 * `target directory name` 
    linked-to directory name as found in the `link search path` during the
    `load`. The value is **ignored** during the `save`.

 * `target file name` 
    linked-to file name, as it would be used to open it, 
    its absolute/relative path, its name and its extension. Path and extensions
    can be empty but the filename cannot.  

 * `target node name` 
    linked-to node name, should be the **absolute** path     
    of the node in the linked-to file.                       

 * `local node name`  
    the **absolute path** of the node in the source Python/CGNS tree. 

The links with a second level file, in other words the links in a file you
are parsing after following a first link, are **always** referred as if you
where in the *target filename*. Then, a list of links can be reused from one
parse to another, because the ``links`` list is relative to the target file.

The example hereafter is a *link-entry*, as an `output` of the `load`
it means the the node ``/Disk/Zone#001/GridCoordinates`` in the file you
were loading is a link the the node ``/Disk#001/Zone#001/Grid#001`` found
in the file ``001disk.cgns``. The file was found in the directory
``/tmp/CGNS-files`` which was in the link search path during the load::

 ['/tmp/CGNS-files','001disk.cgns','/Disk#001/Zone#001/Grid#001','/Disk/Zone#001/GridCoordinates'],

A similar example for the `save`, the *link-entry* would force the node
``/Disk/Zone#001/GridCoordinates`` in the file you are saving to be a link
to the node ``/Disk#001/Zone#001/Grid#001`` of the file ``001disk.cgns``.
There is no check on this target file, the first argument of the entry,
the directory, is ignored.

The directory information is distinct to the filename, because you can
have different actual target files depending on the search paths you set.

.. rubric:: Footnotes

.. [#n1] A Python list is a reference, if you put a list as a child of another
         list the Python interpreter actually refers to the child list. Then
         a child can be shared by two different lists if you do not ask for
         a copy. In other words, the links are the natural way of referencing 
         to lists in Python.

Module details
==============

.. _reference_embedded_map:

Embedded MAP
~~~~~~~~~~~~
The module has is made so that you can take the load/save functions
in order to put them (as C code) in your own application.
This is one of the reason why there is no high level services in this
module.

.. _reference_hdf5:

HDF5
~~~~
The *Hierarchical Data Format* is a data model and its companion library
for low level data storage. The *HDF5* format has replaced *ADF* format
for CGNS low level storage on disks. The required version of *HDF5* is at 
least 1.8.2 which has support for internal symbolic links.

The *MAP* module uses the *CHLone* library instead of the so-called
CGNS/MLL or :term:`CGNS/ADF` libraries.




Glossary
========

.. glossary::

   *cgns.org*
     In this document, *cgns.org* refers to the official CGNS web site and
     by extension to its contents. For example, the *cgns.org* documentation
     is the official documentation found on this web site.

   CGNS/ADF
     The *cgns.org* library implementing the SIDS-to-ADF.

   CGNS/MLL
     The *cgns.org* `mid-level` library, a set of functions providing 
     read/write services for CGNS/SIDS structures. The `MLL` can use
     `ADF` files and `HDF5` files.

.. -------------------------------------------------------------------------


