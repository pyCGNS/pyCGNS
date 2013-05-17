#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from PySide.QtCore       import *
from PySide.QtGui        import *

from CGNS.NAV.Q7HelpWindow import Ui_Q7HelpWindow

# -----------------------------------------------------------------
class Q7Help(QWidget,Ui_Q7HelpWindow):
  def __init__(self,control,key=None,doc=None):
    super(Q7Help, self).__init__()
    self.setupUi(self)
    self._control=control
    if ((doc is None) and (key not in HELPDOCS)): return
    self.eHelp.setAcceptRichText(True)
    self.eHelp.clear()
    if (key is not None):
      self.setWindowTitle("CGNS.NAV: Help on "+HELPDOCS[key][0])
      self.eHelp.insertHtml(HELPDOCS[key][1])
    elif (doc is not None):
      self.setWindowTitle("CGNS.NAV: Contextual Documentation")
      self.eHelp.insertHtml(doc)
    else:
      return
    self.eHelp.moveCursor(QTextCursor.Start,QTextCursor.MoveAnchor)
    self.eHelp.ensureCursorVisible()
    self.eHelp.setReadOnly(True)
  def close(self):
    self._control.help=None
    QWidget.close(self)
  
HELPDOCS={
# --------------------------------------------------------------------------
'Control':('Control panel',
"""
<h2>Control panel</h2>
This main panel of CGNS.NAV is a manager of CGNS/Python tree views, it is 
the first window to be open and and the last to be closed.
You open and close CGNS/Python trees from file with this panel.
The <i>Control panel</i> keeps track of all views you will open in
CGNS.NAV and displays it in a list. A contextual menu on the list allows
you to have short information about the file/tree being viewed and allows some
other basic operations.
<p>
You use the <i>Control panel</i> to manage all the trees you are editing
and to find out which views you have on these trees.

<p>
We use the term <b>view</b>
when a CGNS.NAV window appears in this list
and <b>panel</b> when the window doesn't appear in the list.
Most of the time, a <b>panel</b> is a single instance of a window, while
many <b>view</b>s of the same type can be open at the same time.
<p>
Most views or panels have buttons instead of menus. When you move you mouse
on the button a tooltip appears, you can also use this help. Some button
are not available because the function is not developped yet or because
the context you actually are in doesn't allow to use it.
<p>

When you close the <i>Control panel</i> you leave CGNS.NAV and you close all 
its children windows.

<h3>Table entries</h3>
Each line of the control panel is a view. Each view has a number, which
counts from 001 up to the max number of views you have open. When you close
a view there is no view re-numbering. Some tools, such as the diff or the
merge tools, are using this view number as id to identify an actual
CGNS/Python tree. Many views can be open on the same CGNS/Python tree,
so any view id on the same tree is relevant to identify this tree.
<p>
The status and the type of the view are indicated by an icon. It gives you
information about the ability to read/write the file associated with the
corresponding CGNS/Python tree.
The <b>status</b> changes each time you modify the CGNS/Python tree or
its links or any other parameter that would lead to save a different file
than the one that read at first.
<p>
The <b>status</b> may indicate you cannot save the file, for example if
the file is read only, or if the file has been converted; In this latter
case, the actual file name is a temporary file name. The <b>Info panel</b>
gives you more details about the file itself: right click on a row in the
control panel list and select <b>View Information</b> to get
the <b>Info panel</b>.
<p>
The <b>status</b> icons are:

<table>
<tr><td><img source=":/images/icons/save-S-UM.gif"></td>
<td>File can be saved, no modification to save now</td></tr>
<tr><td><img source=":/images/icons/save-S-M.gif"></td>
<td>File can be saved, some modifications are not saved yet</td></tr>
<tr><td><img source=":/images/icons/save-US-UM.gif"></td>
<td>File cannot be saved, no modification to save now. You have no write access to the file, you should use <b>save as</b> button.</td></tr>
<tr><td><img source=":/images/icons/save-US-M.gif"></td>
<td>File cannot be saved, some modifications are not saved yet. You have no write access to the file, you should use <b>save as</b> button.</td></tr>
</table>
<p>
The second column of a control panel line is the view type, some views can
be open more than once on a given target CGNS/Python tree:

<table>
<tr><td><img source=":/images/icons/tree-load.gif"></td>
<td> Tree view - Open from control panel</td></tr>
<tr><td><img source=":/images/icons/vtkview.gif"></td>
<td> VTK view - Open from tree view</td></tr>
<tr><td><img source=":/images/icons/operate-execute.gif"></td>
<td> Query view - Open from tree view</td></tr>
<tr><td><img source=":/images/icons/form.gif"></td>
<td> Form view - Open from tree view</td></tr>
<tr><td><img source=":/images/icons/operate-list.gif"></td>
<td> Selection view - Open from tree view</td></tr>
<tr><td><img source=":/images/icons/check-all.gif"></td>
<td> Diagnosis view - Open from tree view</td></tr>
<tr><td><img source=":/images/icons/link.gif"></td>
<td> Link view - Open from tree view</td></tr>
</table>

<p>
The third column is the view id. This identifier would be used as
the CGNS/Python tree identification for some CGNS.NAV embedded tools.

<p>
The remaining columns of a control panel line are the directory and the 
filename of the CGNS/Python tree and the target node of the tree. 
These values may be empty if not relevant. For example if you create
a tree from scratch, the directory and the file name are generated and thus
would require a <b>save as</b> to set these values.

All <b>CGNS.NAV</b> windows have a <img source=":/images/icons/top.gif">
button that raises the control panel window. The other way, you select
a view line in the control panel and press <i>[space]</i> to raise the
corresponding view.

You can close a single view, all views related to one tree or all the trees
using the right click menu, on a row.

<h3>Buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/tree-load-g.gif"></td>
<td> load last used file. You can also run CGNS.NAV with the <b>-l</b> option
which has the same effect</td></tr>
<tr><td><img source=":/images/icons/tree-load.gif"></td>
<td> load a file, open the file dialog window. You can give more than one 
filename on the CGNS.NAV command line, each file would be open</td></tr>
<tr><td><img source=":/images/icons/tree-new.gif"></td>
<td> create a new CGNS/Python tree from scratch. There is no associated
filename or directory location, you would have to use <b>save as</b> button
of the <i>Tree view</i>. A default generated filename is set by CGNS.NAV,
but the file is not saved until you ask CGNS.NAV to do so</td></tr>
<tr><td><img source=":/images/icons/pattern.gif"></td>
<td> open the pattern database</td></tr>
<tr><td><img source=":/images/icons/options-view.gif"></td>
<td> open the user options panel</td></tr>
<tr><td><img source=":/images/icons/help-view.gif"></td>
<td> help, you find this help button on almost all windows of CGNS.NAV,
each opens the help page for the current window. You can have only one
help window, then if you open a new one the previous is closed.</td></tr>
<tr><td><img source=":/images/icons/view-help.gif"></td>
<td> about CGNS.NAV and associated modules. You find here the versions
you are currently using</td></tr>
<tr><td><img source=":/images/icons/close-view.gif"></td>
<td> close CGNS.NAV and all its views</td></tr>
</table>

<h3>File and CGNS/Python tree relationships</h3>

The CGNS.NAV tool only uses CGNS/Python tree. When you read a file, all
the CGNS/HDF5 disk data is translated into a CGNS/Python data and the file
is closed. The tool keeps track of the directory and the filename you used
to read, but there is no update if the file changes afterwards.
<p>
In particular if you read a file, modify the CGNS/Python tree and then
save this tree back in the original file, you would <b>overwrite any
other modification that would have been made by another tool meanwhile</b>.
<p>
You can check the status of the related file with the <i>Info panel</i> in
the contextual menu you have by right clicking on a view of the target tree
you want. The <i>Info panel</i> would tell you if the file has changed since
the last time you have read/save it. But this does not guarantee that nothing
would change it between the time you performed this check and the time you
actually save it...
<p>
"""),
# --------------------------------------------------------------------------
'Tree':('Tree view',
"""
<h2>Tree view</h2>

The main window you would use for browsing/editing a CGNS/Python tree is
the <i>Tree view</i>.

<h3>Hierarchical view</h3>
The hierarchical view is a classical tree browser. You click just before the node name to expand/collapse its children node. It supports copy/cut/paste as well
as line editing for node name change, node sidstype change and node value change.
<p>

Bold names are user defined names, the non-bold are names found in the CGNS/SIDS.
<p>
You can use <i>[up][down]</i>arrows to move from a node to another, if you press <i>[control]</i> together with the <i>[up][down]</i> then you move from nodes of a same level.
Pressing <i>[space]</i> sets/unsets the selection flag on the current node.

<h3>Current node and selected nodes</h3>

The current node is the highlighted line. The selected nodes are the lines
with a blue flag on. Some commands require a current node to run, for example
the <i>Form view</i> requires it. Some other commands require a list of
selected nodes, or example the check, or the <i>paste as child for each selected node</i>. This latter command is in the right-click menu.

<h3>Flags</h3>

There a 4 flag columns: <b>L</b> is for link flag, <b>M</b> is mark flag,
<b>C</b> is check flag and the last <b>U</b> is user flag.<br><br>

There are two link flags, <img source=":/images/icons/link.gif"> indicates
the root node of a linked-to node. The actual node is in file used for the
current tree, but the node value and its children are located into another
file. Use the <img source=":/images/icons/link-view.gif"> icon to open the
link view, that shows all existing links for the current tree. Or you can move
the mouse pointer on the link icon, the actual linked-to reference would
be displayed.
The second link flag is
<img source=":/images/icons/link-child.gif"> that indicates the node has a
linked-to ancestor node (its direct parent or any other parent). Depending
on the options you set, you may or may not have the right to change this
child-link node name, type or value.<br><br>

The mark flag is set when you run a selection, or when you mark it by
yourself by pressing <i>[space]</i>. The list of all selected nodes, for
example after running a large query, is displayed using the
<img source=":/images/icons/operate-list.gif"> icon.<br><br>

The check flag is set when you run a check with 
<img source=":/images/icons/check-all.gif">, the diagnosis list is
open with <img source=":/images/icons/check-list.gif">.
Nodes can have a warning <img source=":/images/icons/check-warn.gif">
or an error <img source=":/images/icons/check-fail.gif"> flag. Node without
flags are ok.<br><br>

The last flag column is for the user flag. The user sets/unsets the flag
for each node by selecting the node and pressing one of the <i>[0-9]</i> keys.
Then, by default, the corresponding number is set as flag (for example you
have a <img source=":/images/icons/user-5.gif"> if you press <i>[5]</i>.
The flag is an informative flag, it is not used by CGNS.NAV functions.<br><br>

The snapshot below shows the flag columns with various flag settings.<br><br>
<img source=":/images/icons/help-01.gif"><br>


<h3>Editing nodes</h3>
You can edit node name, type and values by right-cliking on the line/column
you want to edit.<br><br>

The new name is rejected if it already exists or if its syntax is not
compliant with CGNS/SIDS requirements.<br><br>

The new SIDS type should be one of the allowed types.<br><br>

The new value edit depends on its SIDS type and its data type (the *D* column).
You may have an enumerate or a plain string. In case of plain string, you have
to provide a new value as a python value without interpretation.<br><br>

In the case you want to get rid of enumerates, for example when you want
to document a new SIDS type for CPEX proposal, you force plain using
<i>[insert]</i> to edit the name (*NO NAME CHECK*),
<i>[control-insert]</i> to edit the SIDS type and
<i>[shift-insert]</i> to edit value.
These edit methods are not performing any check and your CGNS/Python tree
may be not-compliant, or even impossible to actually store with CGNS/HDF5!

<h3>Copy/Cut/Paste</h3>
The <i>Tree views</i> support the copy/cut/paste of sub-trees.

<h3>Top Buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/save.gif"></td>
<td> save file using the last used file name and save parameters</td></tr>
<tr><td><img source=":/images/icons/tree-save.gif"></td>
<td> save as, as a new directory and/or file name, allows to change save
parameters such as link management</td></tr>
<tr><td><img source=":/images/icons/pattern-save.gif"></td>
<td> save as a pattern (not available yet)</td></tr>
<tr><td><img source=":/images/icons/level-out.gif"></td>
<td> expand one level of children nodes</td></tr>
<tr><td><img source=":/images/icons/level-all.gif"></td>
<td> switch expand all / unexpand all</td></tr>
<tr><td><img source=":/images/icons/level-in.gif"></td>
<td> unexpand one level of children nodes</td></tr>
<tr><td><img source=":/images/icons/flag-all.gif"></td>
<td> set the selection flag on all the nodes. If a node is not visible, then it is flagged as well and all subsequent operation on selection list would take this invisible node as argument. Note that such an invisble node is actually in the CGNS/Python tree but it is not displayed</td></tr>
<tr><td><img source=":/images/icons/flag-revert.gif"></td>
<td> revert nodes selection flag</td></tr>
<tr><td><img source=":/images/icons/flag-none.gif"></td>
<td> remove all selection flags</td></tr>
<tr><td><img source=":/images/icons/operate-list.gif"></td>
<td> open the selection view, gives the list of all selected nodes</td></tr>
<tr><td><img source=":/images/icons/check-all.gif"></td>
<td> check nodes using <b>CGNS.VAL</b> tool. If the selection list is not
empty, then only the selected nodes are checked. If the selection list is
empty, all nodes of the tree are checked.</td></tr>
<tr><td><img source=":/images/icons/check-clear.gif"></td>
<td> remove all check flags</td></tr>
<tr><td><img source=":/images/icons/check-list.gif"></td>
<td> open the diagnosis view which details the check log</td></tr>
<tr><td><img source=":/images/icons/link-select.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/link-add.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/link-delete.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/form-open.gif"></td>
<td> open the form view, gives details on a node including its data</td></tr>
<tr><td><img source=":/images/icons/vtk.gif"></td>
<td> open the VTK view, a 3D view on the actual mesh and its associated data.
This requires a correct parsing of CGNS/Python tree, if the tool is not able
to understand the data (for example if your tree is not CGNS/SIDS compliant,
then no window is open.</td></tr>
<tr><td><img source=":/images/icons/plot.gif"></td>
<td> open the Plot view, a 2D view on some tree data.
Same remarks as VTK view.</td></tr>
<tr><td><img source=":/images/icons/pattern-view.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/link-view.gif"></td>
<td> open the Link view, gives details on the links</td></tr>
<tr><td><img source=":/images/icons/operate-view.gif"></td>
<td> open the Query view, a powerful mean to edit and run Python commands on the tree</td></tr>
<tr><td><img source=":/images/icons/check-view.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/tools-view.gif"></td>
<td> not available yet</td></tr>
<tr><td><img source=":/images/icons/snapshot.gif"></td>
<td> creates a bitmap file with the snapshot of the current tree view.
See user options for directory and file used for this snapshot.</td></tr>
</table>

"""),
# --------------------------------------------------------------------------
'VTK':('VTK view',
"""
<h2>VTK view</h2>

<h3>Selection list</h3>
<p>
<table>
<tr><td><img source=":/images/icons/selected.gif"></td>
<td>The selection list entry is selected for operations</td></tr>
<tr><td><img source=":/images/icons/unselected.gif"></td>
<td>The selection list entry is NOT selected for operations</td></tr>
<tr><td><img source=":/images/icons/hidden.gif"></td>
<td>The selection list entry is hidden, the data is still there but it is not
displayed in the window.</td></tr>
</table>

<h3>Icons and buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/zoompoint.gif"></td>
<td>Switch to the zoom-window mode. You draw a window using the mouse left
button and the zoom would fit to the defined window.</td></tr>
<tr><td><img source=":/images/icons/zoom-actor.gif"></td>
<td>Reset the view to fit the object in the window</td></tr>
<tr><td><img source=":/images/icons/lock-legend.gif"></td>
<td>Switch to move legend mode, used to mose the reference axis draw, the
legend and other extra information displayed on the view.</td></tr>
<tr><td><img source=":/images/icons/value.gif"></td>
<td>Switch to the show value mode, in the case you have actual values,
the pick of a node would show corresponding value.</td></tr>
<tr><td><img source=":/images/icons/colors.gif"></td>
<td>Randomly change the colors of view objects.</td></tr>
<tr><td><img source=":/images/icons/colors-bw.gif"></td>
<td>Switch the view background to balck/white.</td></tr>
</table>


<h3>Displaying actual values</h3>
<p>


<h3>User defined camera views</h3>
<p>
You can the current camera and reset to this stored point of view later on.
<table>
<tr><td><img source=":/images/icons/camera-snap.gif"></td>
<td>Sotre an existing camera</td></tr>
</table>


<h3>Interactions</h3>

<h4>Mouse buttons</h4>
The <i>mouse left button</i> runs the rotation (see below). The
<i>mouse right button</i> runs the zoom, same with the <i>mouse wheel</i>
if you have one. The <i>mouse middle button</i>, if you have one, runs a
translation parallel to the view plane. 

<h4>Rotation</h4>
The <i>Camera axis all</i> uses the center of the view (the center of the
camera view) as the center of rotation.<br>
The <i>Object axis all</i> is a rotation with the center of the object
as rotation center.<br>
The <i>Object axis x</i> mode forces a rotation with x as single axis, same
for y and z.<br>

<h4>Key Bindings</h4>
<b>s</b> Surface mode rendering<br>
<b>w</b> Wire mode rendering<br>
<b>q</b> Surface and wire mode rendering<br>
<b>r</b> Fit view to object<br>

<b>z</b> Select a Zone<br>
<b>p</b> Pick a point in the selected Zone<br>
"""),
# --------------------------------------------------------------------------
'Link':('Link view',
"""
<h2>Link view</h2>

<h3>Buttons</h3>
<p>

"""),
# --------------------------------------------------------------------------
'Form':('Form view',
"""
<h2>Form view</h2>

This view shows all information it can find about a node. The <i>Form view</i>
displays the node and its data using different ways and allows you some
basic operations on this node. Each tab is a view of the same node. The
<i>table tab</i> is a raw view of dimensions, types and shows the
actual <i>data</i> in a spreadsheet-like widget.


<h3>Spreadsheet</h3>
<p>
The data is displayed in a 2D table, you can change the horizontal/vertical
distribution using the <i>Size</i> selector. It computes for you all
possible combinations of horizontal and vertical indicies from the original
data size.

<h3>Links</h3>
<p>
All details about relationship with other files are in the <i>Link</i> tab.
It shows you the current link if the node is actually the root node of a link,
it also shows you the parent node link if the current node is the child of a
linked-to node. In other words, it tells you wether the node you are looking at
is in the top file or not.

<h3>Text</h3>
<p>
Some node data are text, it is often more readable to look at it using a real
text tool than a spreadsheet-like tool. The <i>Text</i> tab is available only
for text data.

"""),
# --------------------------------------------------------------------------
'Query':('Query view',
"""
<h3><font color=red>WARNING</font></h3>
The queries can modify your CGNS/Python tree in such a way that it could
become non-compliant, inconsistant in CGNS/Python or even <font color=red>
<b>crash</b> the CGNS.NAV application</font>
or any of its sub-applications such as MAP or VAL.<p>
<font color=red>Use of CGNS/Python data modification requires a very
good level of Python and CGNS skills, use at your own risks...</font>

<h2>Query view</h2>
The query view is one of CGNS.NAV most powerful feature. It provides a
true Python scripting access on the CGNS tree, the script you write is
applied to each node of the whole CGNS/Python tree. You have to write the
script using the usual Python syntax and semantics, including the required
<b>import</b>s statements, however CGNS.NAV adds a pre-script and a post-script
to your actual text.<br><br>

The pre-script is used to defined local variable that would be useful for you.
The variables are setting the node context, you can get/set these values as
these are the actual node:

<table>
<tr><td><b>NODE</b></td><td>the NODE as a CGNS/Python NODE, that is a list
with <i>[ NAME, VALUE, CHILDREN, SIDSTYPE]</i></td></tr>
<tr><td><b>NAME</b></td><td>the NAME of the NODE, same as NODE[0]</td></tr>
<tr><td><b>VALUE</b></td><td>the VALUE of the NODE, same as NODE[1]</td></tr>
<tr><td><b>SIDSTYPE</b></td><td>the CGNS/SIDS type of the NODE, same
as NODE[3]</td></tr>
<tr><td><b>CHILDREN</b></td><td>the list of CHILDREN of the NODE,
same as NODE[2]</td></tr>
<tr><td><b>PARENT</b></td><td>the PARENT node of current NODE</td></tr>
<tr><td><b>TREE</b></td><td>the complete CGNS/Python TREE</td></tr>
<tr><td><b>PATH</b></td><td>the PATH to the current NODE</td></tr>
<tr><td><b>ARGS</b></td><td>The arguments tuple you may have passed (in
the Tree view for example). Please note this is always a tuple, even if you
have a single argument (then use ARGS[0]). The is a special case, if your
argument is a single string then you need not to put quotes around. For
example, you can use <i>ZoneType</i> instead of <i>'ZoneType'</i>.</td></tr>
<tr><td><b>RESULT</b></td><td>the output of your script, this result value
is inserted into the global result list for all nodes. Thus, you would
rather add a tuple containing the current PATH and the result if you
want to find back which result matches which value (see example)</td></tr>
</table>

<h3>Buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/operate-add.gif"></td>
<td>Add a new query</td></tr>
<tr><td><img source=":/images/icons/operate-delete.gif"></td>
<td>Remove an existing query</td></tr>
<tr><td><img source=":/images/icons/operate-save.gif"></td>
<td>Save all queries</td></tr>
</table>

"""),
# --------------------------------------------------------------------------
'Option':('Option panel',
"""
<h2>Option panel</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'File':('File panel',
"""
<h2>File panel</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Selection':('Selection view',
"""
<h2>Selection view</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Diagnosis':('Diagnosis view',
"""
<h2>Diagnosis view</h2>

A <i>Diagnosis view</i> is associated to one <i>Tree view</i> once a check
of its CGNS/Python tree has been performed. The <i>Diagnosis view</i>
shows the errors and warnings per node, filter the error/warning type you
want to see or suppress the warnings display. You can browse these diagnosis
and go back to the targeted node in the <i>Tree view</i> by pressing the
<i>[space]</i> key.

<h3>Filters</h3>
<p>
You can select an error/warning code in the combo-box, it shows all the
error/warning codes the check has collected. In the case you have too
many warnings, you can just ignore them by un-setting the <i>Warnings</i>
checkbox.

<h3>Buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/level-in.gif"></td>
<td> expand all errors/warnings</td></tr>
<tr><td><img source=":/images/icons/level-out.gif"></td>
<td> collapse all errors/warnings</td></tr>
<tr><td><img source=":/images/icons/node-sids-opened.gif"></td>
<td> go to previous filtered error/warning</td></tr>
<tr><td><img source=":/images/icons/selected.gif"></td>
<td> go to next filtered error/warning</td></tr>
<tr><td><img source=":/images/icons/check-save.gif"></td>
<td> save errors/warnings as a Python importable file (see hereafter)</td></tr>
</table>

<h3>Save diagnosis</h3>
<p>
You can save all the diagnosis in a file. The file contains a Python
dictionnary in the variable <i>data</i> with the node paths as key.
For a key, the value is the list of diagnosis, each diagnosis is a tuple
of three strings: the error/warning code, the level, the message.
Here is an example of such a file:
<pre>
data={
'/SquaredNozzle/INJ3/.Solver#Trigger/next_iteration':
  ("S004","E","DataType [I4] not allowed for this node"),
'/SquaredNozzle/INJ3/.Solver#Trigger/next_state':
  ("S004","E","DataType [I4] not allowed for this node")
}
</pre>

"""),
# --------------------------------------------------------------------------
'Info':('Info panel',
"""
<h2>Info panel</h2>

Gives all details on the top file you use to load/save the target CGNS/Tree.


<b>In case of links, only the top file is detailled.</b>

<h3>Translated files</h3>
<p>

As <i>pyCGNS</i> uses only GCNS/HDF5 files, using <i>CHLone</i>, the CGNS/ADF
files are translated on the fly when you load them. The translation tool is
<i>cgnsconvert</i> and its location should be set into the <i>Option panel</i>.

A translated file is stored into a temporary directory, the <i>Info panel</i>
shows you which is the actual original file name and the temporary filename.
"""),
# --------------------------------------------------------------------------
'Pattern':('Pattern panel',
"""
<h2>Pattern panel</h2>

"""),
# --------------------------------------------------------------------------
'Tools':('Tools panel',
"""
<h2>Tools panel</h2>

"""),
# --------------------------------------------------------------------------
}

# --- last line
