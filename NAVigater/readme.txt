.. -------------------------------------------------------------------------
.. pyCGNS - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.NAV
========

If you want to browse your CGNS file, just type::

  CGNS.NAV

and the pyCGNS browser appears. 

.. image:: ../doc/images/cgnsnav_overview.png
   :width: 26cm

.. image:: ../doc/images/cgnsnav_01.png
   :width: 15cm
.. image:: ../doc/images/cgnsnav_02.png
   :width: 15cm
.. image:: ../doc/images/cgnsnav_03.png
   :width: 20cm
.. image:: ../doc/images/cgnsnav_04.png
   :width: 20cm
.. image:: ../doc/images/cgnsnav_05.png
   :width: 15cm

The file format is detected using the file extension. You can give a list
of files to ``CGNS.NAV``, it opens each file w.r.t. its type.
The ``.cgns`` extension uses the CGNS library which can detect both
CGNS/ADF and CGNS/HDF formats::

  CGNS.NAV wing.cgns plane.adf helicopter.hdf cror.py

It can load/save CGNS files with
the HDF5/ADF and Python formats, parse and display the contents,
edit the contents using simple edit and copy/paste commands, select
nodes using complex queries, use already defined patterns such as the *SIDS*
patterns and other interoperability features.

-----

.. toctree::

   quickstart
   treeView
   optionView
   patternView
   vtkView
   queryView
   linkView
   tableView

-----

.. warning::

   There are a *lot* of screenshots in this ``CGNS.NAV`` doc, some may
   be a bit out-dated but most of the look-and-feel of the tool would
   keep unchanged.

.. include:: ../doc/Intro/glossary.txt

.. _nav_index:

NAV Index
---------

* :ref:`genindex`

.. -------------------------------------------------------------------------
