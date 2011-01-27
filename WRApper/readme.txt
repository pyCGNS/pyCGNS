.. -------------------------------------------------------------------------
.. pyCGNS - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.WRA
========

The *CGNS/MLL* and *CGNS/ADF* wrapper. Provides the user with a *Python*-like
interface to the *CGNS/MLL* and *CGNS/ADF* functions. There are two classes
acting as proxies to the *MLL* and the *ADF* calls. The interface to the calls
are most of the time unchanged, that is the order and the types of the 
arguments are more or less the same as for the C/Fortran APIs.

The file id is always the first argument in the C/Fortran API, it is
replaced by the class itself which holds the required connection information.

.. warning::

   The so-called *ADF* calls of the **v3** version of *CGNS* libraries 
   can read/write *ADF* as well as *HDF5* 
   files (see :ref:`File formats <wrainter:v3multifileformat>`).

Examples
--------

The following example shows how to create links. A ``secondfile`` is open
with the *ADF* interface and then we add links to the ``firstfile`` grid.
This is a typical pattern used to share a large grid between several files::

   import CGNS
   import CGNS.WRA._adf as ADF

   firstfile="M6grid.cgns"
   secondfile="M6comput.cgns"

   filesol=CGNS.pyADF(secondfile,ADF.OLD,ADF.NATIVE)
   for bn in filesol.children_names(filesol.root()):
     bx=filesol.get_node_id(filesol.root(),bn)
     if (filesol.get_label(bx) == 'CGNSBase_t'):
       for zn in filesol.children_names(bx):
        zx=filesol.get_node_id(bx,zn)
        if (filesol.get_label(zx) == 'Zone_t'):
         filesol.link(zx,'GridCoordinates',firstfile,'/%s/%s/GridCoordinates'%(bn,zn))
   filesol.database_close()

The next example uses the *CGNS/MLL* calls. Again we are creating links from
the ``secondfile`` to the ``firstfile``, or more exactly from a list of
``secondfiles`` to the ``firstfile``. The purpose of the script is to gather 
a set of *CGNS* files produced by a parallel computation, the ``secondfile``
is an empty skeletton with links to the actual files.

Contents
--------

.. toctree::
   
   API
   fileformats

-----

.. _wra_index:

WRA Index
---------

* :ref:`genindex`

.. -------------------------------------------------------------------------
