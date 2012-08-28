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
   can read/write *ADF* as well as *HDF5* files.

MLL Examples
------------

If you really want to use CGNS/MLL, for example if you have a large toolbox 
with old pyCGNS scripts you have to change your
imports from `CGNS.WRA` to `CGNS.WRA.wrapper` for functions. There is a
long example with fake data, the purpose here is to show the syntax::

    import CGNS.WRA.wrapper
    import CGNS.PAT.cgnskeywords as CGK
    import numpy as NPY

    db=CGNS.WRA.wrapper.pyCGNS('testfile.cgns',CGK.MODE_WRITE)
    db.basewrite('Base',3,3)
    db.zonewrite(1,'Zone 01',[3,5,7,2,4,6,0,0,0],CGK.Structured)
    db.zonewrite(1,'Zone 02',[3,5,7,2,4,6,0,0,0],CGK.Structured)

    # beware: the keywords for strings have the _s postfix like below
    #         but enumerates are used without postfix
    #
    db.coordwrite(1,1,CGK.RealDouble,CGK.CoordinateX_s,c01[0])
    db.one2onewrite(1,1,"01-02","Zone 02",(3,1,1,3,5,7),(1,1,1,1,5,7),(1,2,3)) 
    db.bcwrite(1,1,"I low",CGK.BCTunnelInflow,CGK.PointRange,[(1,1,1),(3,2,4)])
    db.bcdatasetwrite(1,1,1,"I low DATA SET",CGK.BCTunnelInflow)
    db.bcdatawrite(1,1,1,1,CGK.Neumann)

    # we have now nodes that can be set as children of different node types,
    # then we have to set the parent node using the goto. The current parent
    # is a variable for the current CGNS file.
    #
    db.goto(1,[])
    db.statewrite(CGK.ReferenceState_s)

    db.goto(1,[(CGK.ReferenceState_t,1)])
    db.arraywrite("Mach",CGK.RealDouble,1,(1,),NPY.array([0.8],'d'))

    db.close()

When a function call fails, it sets the error code and error message. You
have to check it at each call if you want to make sure there is no problem::

    import CGNS.WRA.wrapper
    import CGNS.PAT.cgnskeywords as CGK

    db=CGNS.WRA.wrapper.pyCGNS('testfile.cgns',CGK.MODE_WRITE)
    if (db.error[0] != 0):
      print "# Error code:%d message:[%s]"%(db.error[0],db.error[1])

The actual data is passed to the MLL using the numpy arrays. When you write
you have to create the numpy array, once it is passed to the function you
can delete it::

  w=NPY.array([[0.891,4.12],[1.0,2.2],[3.14159,3.2]],'d')
  db.arraywrite("OriginLocation",CGK.RealDouble,len(w.shape),w.shape,w)
 
When you read, the MLL returns a new numpy array to you, this array now
belongs to you and you can close the CGNS file and/or delete the array, there
is no more relationship between the array and MLL::

  r=db.fieldread(1,2,2,CGK.Density_s,CGK.RealDouble,[1,1,1],[2,4,6])
  db.close()
  print r.shape

ADF Examples
------------

The following example shows how to create links. A ``secondfile`` is open
with the *ADF* interface and then we add links to the ``firstfile`` grid.
This is a typical pattern used to share a large grid between several files::

   import CGNS.WRA.wrapper
   import CGNS.WRA._adf as ADF

   firstfile="M6grid.cgns"
   secondfile="M6comput.cgns"

   filesol=CGNS.WRA.wrapper.pyADF(secondfile,ADF.OLD,ADF.NATIVE)
   for bn in filesol.children_names(filesol.root()):
     bx=filesol.get_node_id(filesol.root(),bn)
     if (filesol.get_label(bx) == 'CGNSBase_t'):
       for zn in filesol.children_names(bx):
        zx=filesol.get_node_id(bx,zn)
        if (filesol.get_label(zx) == 'Zone_t'):
         filesol.link(zx,'GridCoordinates',
                      firstfile,'/%s/%s/GridCoordinates'%(bn,zn))
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
