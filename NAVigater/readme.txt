.. -------------------------------------------------------------------------
.. pyCGNS - CFD General Notation System - 
.. See license.txt file in the root directory of this Python module source  
.. -------------------------------------------------------------------------

CGNS.NAV
========

If you want to browse your CGNS file, just type::

  CGNS.NAV

and the following control view appears:

.. image:: ../doc/images/cgnsnav_im01.png
   :width: 18cm

.. |logo1| image:: ../doc/images/cgnsnav_im02.png 

.. |logo2| image:: ../doc/images/cgnsnav_im03.png 


.. |logo1b| image:: ../doc/images/cgnsnav_im53.png 

To load a new CGNS file, you must click the |logo1|, select the wanted CGNS file by typing its name and by
clicking the |logo1b| icon.  But you can also reload the last used CGNS file by clicking on the icon |logo2|,
and the tree view appears:

.. image:: ../doc/images/cgnsnav_im04b.png
   :width: 14cm

.. |logo3| image:: ../doc/images/cgnsnav_im05.png 

.. |logo2b| image:: ../doc/images/cgnsnav_im43.png 

.. |logo3b| image:: ../doc/images/cgnsnav_im44.png 

.. |logo4b| image:: ../doc/images/cgnsnav_im45.png 

.. |logo5b| image:: ../doc/images/cgnsnav_im46.png 

.. |logo6b| image:: ../doc/images/cgnsnav_im47.png 

.. |logo7b| image:: ../doc/images/cgnsnav_im48.png

.. |logo8b| image:: ../doc/images/cgnsnav_im49.png

.. |logo9b| image:: ../doc/images/cgnsnav_im50.png

.. |logo10b| image:: ../doc/images/cgnsnav_im51.png

.. |logo11b| image:: ../doc/images/cgnsnav_im52.png


As you can observe it, there is only one entry in the tree view. This is the root of our CGNS file
which can contain one or several bases.The opening of the CGNS tree occurs in a recursive way.
If you want to expand this node one level up in order to display the entries corresponding to the base 
contained in the ``CGNSTree`` node, click the |logo4b| icon.  

If you perform this operation again with the ``base1`` node, you obtain the following tree view:

.. image:: ../doc/images/cgnsnav_im54.png

You can repeat the operation for the nodes of the different zones which are under the ``base1`` node.
And so on...

In order to expand the tree view one level down, click the |logo2b| icon.

To expand all the loaded CGNS/tree, you must click |logo3|.

You can then see the expanded CGNS/tree:

.. image:: ../doc/images/cgnsnav_im06b.png 
    :width: 17cm

If you only want to expand or collapse individual nodes in the tree, you double-click on the name of
the desired node.



*To deal with the tree view*

+-----------------+---------------------------------------------------------------------------+
|   Icon          | Action								      |
+=================+===========================================================================+
| |logo2b|        | Expand the tree one level down.                                           |
+-----------------+---------------------------------------------------------------------------+
| |logo3b|        | Expand all the tree.                                                      |
+-----------------+---------------------------------------------------------------------------+
| |logo4b|        | Expand the tree one level up.                                             |
+-----------------+---------------------------------------------------------------------------+
| |logo5b|        | Select the previous marked node.                                          |
+-----------------+---------------------------------------------------------------------------+
| |logo6b|        | Select the next marked node.                                              |
+-----------------+---------------------------------------------------------------------------+
| |logo7b|        | Switch between the marked nodes and the unmarked nodes.                   |
+-----------------+---------------------------------------------------------------------------+
| |logo8b|        | Unmark all the nodes.                                                     |
+-----------------+---------------------------------------------------------------------------+
| |logo9b|        | Mark all the nodes.                     .                                 |
+-----------------+---------------------------------------------------------------------------+
| |logo10b|       | Open the selected nodes list.                                             |
+-----------------+---------------------------------------------------------------------------+
| |logo11b|       | Display the mesh of the tree.                                             |
+-----------------+---------------------------------------------------------------------------+

.. |logo4| image:: ../doc/images/cgnsnav_im07.png
 
To display the mesh, element sets, connectivities and boundary conditions contained in the CGNS file, 
click on |logo4|.

and the mesh is dispayed in this window:

.. image:: ../doc/images/cgnsnav_im08.png 
    :width: 20cm

The view can be translated, rotated and scaled by using the mouse. The three axis x,y,z are displayed in the 
lower left corner of the window. The x-axis is coloured in red, the y-axis in yellow and the z-axis in green.

 *Mouse Bindings*

The mouse bindings and the related actions are:

 +-----------------+---------------------------------------------------------------------------+
 |   Button        | Action								       |
 +=================+===========================================================================+
 | ``Button 1``    | Rotate the camera around its focal point.                                 |
 +-----------------+---------------------------------------------------------------------------+
 | ``Button 2``    | Translate the elements displayed in the window.                           |
 +-----------------+---------------------------------------------------------------------------+
 | ``Button 3``    | Adjust the view by holding down this button while moving                  |
 |                 | the mouse in the display window. The objects are scaled up when the       |
 |                 | mouse moves from bottom to up and they are scaled down when the mouse     |
 |                 | moves from up to bottom.                                                  |
 +-----------------+---------------------------------------------------------------------------+

To select an element of the CGNS/tree, you perform a pick operation by positioning the mouse 
cursor on the place of your choice and by pressing on the ``p`` key.

.. image:: ../doc/images/cgnsnav_im11.png 
    :width: 17cm

The pick operation shoots a ray into the 3D scene and returns information about the objects that 
the ray hits. The first element hited by the ray is highlighted in red and a blue wireframe outlines
the bounding box of the selected object.In the left upper corner, the list of the paths of elements
hited by the ray appears. In our case, there is only one path because only one object was hited by the shot ray. 
The path of the selected element also appears in the box. In this example, the object's path is 
``dom1/PENTA_6{TRI}``.

.. image:: ../doc/images/cgnsnav_im12.png 

.. |logo5| image:: ../doc/images/cgnsnav_im13.png 

.. |logo6| image:: ../doc/images/cgnsnav_im14.png 


You can see that the paths of selected elements are marked by the icon |logo5|
while the unselected elements are marked by the icon |logo6|. 


.. |logo7| image:: ../doc/images/cgnsnav_im17.png 

.. |logo8| image:: ../doc/images/cgnsnav_im18.png 

.. |logo9| image:: ../doc/images/cgnsnav_im19.png 

.. |logo10| image:: ../doc/images/cgnsnav_im20.png 

*Camera's position*

+-------------+-------------------------------------------------------------------------------+
|  Icon       | Action                                                                        |
+=============+===============================================================================+
| |logo7|     | Set the camera along the -X axis.                                             |
+-------------+-------------------------------------------------------------------------------+
| |logo8|     | Set the camera along the -Y axis.                                             |
+-------------+-------------------------------------------------------------------------------+
| |logo9|     | Set the camera along the -Z axis.                                             |
+-------------+-------------------------------------------------------------------------------+
| |logo10| +  | Set the camera along the +X, +Y or +Z axis if the mirror case is checked.     |
| |logo7|     |                                                                               |
+-------------+-------------------------------------------------------------------------------+
| ``Ctrl`` +  | Rotate about the X,Y or Z direction.                                          |
|  |logo7|    |                                                                               |
+-------------+-------------------------------------------------------------------------------+

If you want to set the viewing position to view the data along -Z axis, just click on |logo9|.
To display the view along the opposite direction, +Z axis, check |logo10| and click on |logo9|.

.. image:: ../doc/images/cgnsnav_im21.png 
   :width: 10cm

.. image:: ../doc/images/cgnsnav_im22.png 
   :width: 11cm
 
To display the other elements of the CGNS file, you have to handle the view with the mouse.
By using the ``Button 1`` of the mouse to rotate the view, the ``Button 2`` to tranlate it and
the ``Button 3`` with a motion of the mouse from bottom to up to scale up the elements, the view is 
adjusted like that:

.. image:: ../doc/images/cgnsnav_im55.png 
    :width: 17cm

and you perform a pick operation by pressing the ``p`` key:

.. image:: ../doc/images/cgnsnav_im56.png 
    :width: 17cm

All the paths of the selected objects are displayed in the left upper corner of the window and
they are marked with the |logo5| in the list below. As you can see, the current selected object 
is the same as previously, namely ``dom1/PENTA_6{TRI}`` because it's the closest object to the camera.
 
.. |logo11| image:: ../doc/images/cgnsnav_im25.png 

.. |logo12| image:: ../doc/images/cgnsnav_im26.png 

.. |logo13| image:: ../doc/images/cgnsnav_im27.png 

.. |logo14| image:: ../doc/images/cgnsnav_im28.png 

If a particular view interests you, you can save it and restore it later.
When a desired view is achieved, type a view's name in the box |logo14| located at the top of the 
window and press the ``Enter`` key or click the |logo11| icon to save the view. 

You can add a number of different views by repeating the previous operation.

Know, you want to have an overall view of the tree. To do that, press on the ``r`` key.

.. image:: ../doc/images/cgnsnav_im57.png 
    :width: 16cm

When you pressed the ``r`` key, the size of the objects changed to fit in the vtk view, the objects
are centered while the camera keeps the current view direction.

To restore a saved view, choose the view's name in the list, like below: 

.. image:: ../doc/images/cgnsnav_im29.png 

To delete an unwanted view, select the view's name and click the |logo13| icon. 

*To save a view*

+-------------+-------------------------------------------------------------------------------+
|  Icon       | Action                                                                        |
+=============+===============================================================================+
| |logo11|    | Save the current view and add it to the view list.                            |
+-------------+-------------------------------------------------------------------------------+
| |logo12|    | Write the view list into a file.                                              |
+-------------+-------------------------------------------------------------------------------+
| |logo13|    | Remove the current view from the view list.                                   |
+-------------+-------------------------------------------------------------------------------+
| |logo14|    | Type here the view's name to save.                          	              |
+-------------+-------------------------------------------------------------------------------+


.. |logo15| image:: ../doc/images/cgnsnav_im32.png 

.. |logo16| image:: ../doc/images/cgnsnav_im33.png 

You can change the colours randomly by clicking the |logo16| icon and by clicking the |logo15| icon, 
you switch between a black background and a white background, like below:

.. image:: ../doc/images/cgnsnav_im34.png 

*To change colours*

+-------------+-------------------------------------------------------------------------------+
|  Icon       | Action                                                                        |
+=============+===============================================================================+
| |logo16|    | Change the colours randomly.                                                  |
+-------------+-------------------------------------------------------------------------------+
| |logo15|    | Switch between a black background and a white background.                     |
+-------------+-------------------------------------------------------------------------------+

.. |logo17| image:: ../doc/images/cgnsnav_im35.png 

.. |logo18| image:: ../doc/images/cgnsnav_im36.png

.. |logo19| image:: ../doc/images/cgnsnav_im37.png

.. |logo20| image:: ../doc/images/cgnsnav_im41.png

You can also modify the selected object by using the |logo17| icon and the |logo18| icon.


.. image:: ../doc/images/cgnsnav_im38.png


When you perform this operation, the next element |logo17| or the previous element |logo18| of the 
list which contains the objects picked becomes the current selected object. After the last element 
of the selection is reached, the first object of the list is again selected as current selected element. 
 
To unselect all elements, click the |logo19| icon.

*To modify the selected object*

+-------------+----------------------------------------------------------------------------------------+
|  Icon       | Action										       |
+=============+========================================================================================+
| |logo17|    | Change the selected object by taking the following object in the selected objects list.|
+-------------+----------------------------------------------------------------------------------------+
| |logo18|    | Change the selected object by taking the previous object in the selected objects list. |   
+-------------+----------------------------------------------------------------------------------------+
| |logo19|    | Set all elements as unselected objects and the hidden objects become visible.          |
+-------------+----------------------------------------------------------------------------------------+

To see better a part of the view, you can remove visible elements of the tree. Once you performed 
a pick operation to select objects, press the ``d`` key to hide the current selected element. After the objet
is hidden, the next element located in the selection list becomes the current selected element. You can repeat 
the operation as long as list of the selected objects isn't empty.

.. image:: ../doc/images/cgnsnav_im39.png 


As you can observe it, the hidden objects are marked with the |logo20| icon.

.. image:: ../doc/images/cgnsnav_im40.png 

When you click the |logo19| icon, all elements become unselected and the objects which are hidden
become again visible. Consequently, all elements of the list are marked with |logo6| icon.



*Key Bindings* 

The following keys and the corresponding actions are:

 +----------+----------------------------------------------------------------------------------+
 |  Key     | Action                                                                           |
 +==========+==================================================================================+
 | ``f``    | Fly to the picked point.                                                         |
 +----------+----------------------------------------------------------------------------------+
 | ``p``    | Perform a pick operation.                                                        |
 +----------+----------------------------------------------------------------------------------+
 | ``r``    | The elements are centered and the camera moves along the current                 |
 |          | view direction so that all elements are visible in the window  .                 |
 +----------+----------------------------------------------------------------------------------+
 | ``s``    | Modify the representation of all elements so that they are surfaces.             |
 +----------+----------------------------------------------------------------------------------+
 | ``w``    | Modify the representation of all elements so that they are wireframes.           |              
 +----------+----------------------------------------------------------------------------------+
 | ``d``    | Hide the current element selected by performing a pick operation.                |
 +----------+----------------------------------------------------------------------------------+
 | ``Ctrl`` | Add the elements selected by a pick operation to the previous selection.         |
 +----------+----------------------------------------------------------------------------------+


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
