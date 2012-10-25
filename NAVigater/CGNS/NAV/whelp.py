#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from PySide.QtCore       import *
from PySide.QtGui        import *

from CGNS.NAV.Q7HelpWindow import Ui_Q7HelpWindow

# -----------------------------------------------------------------
class Q7Help(QWidget,Ui_Q7HelpWindow):
  def __init__(self,control,key=None):
    super(Q7Help, self).__init__()
    self.setupUi(self)
    self._control=control
    if (key not in HELPDOCS): return
    self.setWindowTitle("CGNS.NAV: Help on "+HELPDOCS[key][0])
    self.eHelp.setAcceptRichText(True)
    self.eHelp.clear()
    self.eHelp.insertHtml(HELPDOCS[key][1])
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
The main panel of CGNS.NAV is a summary of views. The list gives you
short information about the file/tree being viewed in each window.
<p>
You can open new trees, close views or close all CGNS.NAV from this panel.

<h3>Table entries</h3>
Each line of the control panel is a view. Each view has a number, which
counts from 001 up to the max number of views you have. The status and the
type of the view are indicated by an icon.
<p>

The <b>status</b> changes each time you modify the CGNS/Python tree or
its links or any other parameter that would lead to save a different file
than the one that was loaded at first.

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
The second column of a control panel line is the view type:

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
The remaining columns of a control panel line are the directory and the 
filename of the CGNS/Python tree and the target node of the tree. 
These values may be empty if not relevant. For example if you create
a tree from scratch, the directory and the file name are generated and thus
would require a <b>save as</b> to set these values.

All <b>CGNS.NAV</b> windows have a <img source=":/images/icons/top.gif">
button that raises the control panel window. The other way, you select
a view line in the control panel and press <i>[space bar]</i> to raise the
corresponding view.

You can close a single view, all views related to one tree or all the trees
using the right click menu, on a row.

<h3>Buttons</h3>
<p>
<table>
<tr><td><img source=":/images/icons/tree-load-g.gif"></td>
<td> load last used file</td></tr>
<tr><td><img source=":/images/icons/tree-load.gif"></td>
<td> load a file, open the file dialog window</td></tr>
<tr><td><img source=":/images/icons/tree-new.gif"></td>
<td> create a new CGNS/Python tree from scratch</td></tr>
<tr><td><img source=":/images/icons/pattern.gif"></td>
<td> open the pattern database</td></tr>
<tr><td><img source=":/images/icons/option-view.gif"></td>
<td> open the user options panel</td></tr>
<tr><td><img source=":/images/icons/help-view.gif"></td>
<td> help</td></tr>
<tr><td><img source=":/images/icons/view-help.gif"></td>
<td> about CGNS.NAV</td></tr>
<tr><td><img source=":/images/icons/close-view.gif"></td>
<td> close CGNS.NAV and all its views</td></tr>
</table>

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
<td> open the selection panel, gives the list of all selected nodes</td></tr>
<tr><td><img source=":/images/icons/check-all.gif"></td>
<td> check nodes using <b>CGNS.VAL</b> tool. If the selection list is not
empty, then only the selected nodes are checked. If the selection list is
empty, all nodes of the tree are checked.</td></tr>
<tr><td><img source=":/images/icons/check-clear.gif"></td>
<td> remove all check flags</td></tr>
<tr><td><img source=":/images/icons/check-list.gif"></td>
<td> open the diagnosis panel which details the check log</td></tr>
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

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">

<h3>Key Bindings</h3>
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
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Form':('Form view',
"""
<h2>Form view</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Query':('Query view',
"""
<h2>Query view</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
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
'Selection':('Selection panel',
"""
<h2>Selectione panel</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Diagnosis':('Diagnosis panel',
"""
<h2>Diagnosis panel</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
'Info':('Info panel',
"""
<h2>Info panel</h2>

<h3>Buttons</h3>
<p>
<img source=":/images/icons/unselected.gif">
"""),
# --------------------------------------------------------------------------
}

# --- last line
