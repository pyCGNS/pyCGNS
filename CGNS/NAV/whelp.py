#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#

from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QTextCursor

from CGNS.NAV.Q7HelpWindow import Ui_Q7HelpWindow


# -----------------------------------------------------------------
class Q7Help(QWidget, Ui_Q7HelpWindow):
    def __init__(self, control, key=None, doc=None):
        super(Q7Help, self).__init__()
        self.setupUi(self)
        self._control = control
        if (doc is None) and (key not in HELPDOCS):
            return
        self.eHelp.setAcceptRichText(True)
        self.eHelp.clear()
        if key is not None:
            self.setWindowTitle("CGNS.NAV: Help on " + HELPDOCS[key][0])
            self.eHelp.insertHtml(HELPDOCS[key][1])
        elif doc is not None:
            self.setWindowTitle("CGNS.NAV: Contextual Documentation")
            self.eHelp.insertHtml(doc)
        else:
            return
        self.eHelp.moveCursor(QTextCursor.Start, QTextCursor.MoveAnchor)
        self.eHelp.ensureCursorVisible()
        self.eHelp.setReadOnly(True)

    def close(self):
        self._control.help = None
        QWidget.close(self)


HELPDOCS = {
    # --------------------------------------------------------------------------
    "Control": (
        "Control panel",
        """
                <h2>Control panel</h2>
                The main panel of CGNS.NAV is a summary of views. The list gives you
                short information about the file/tree being viewed and allows you some
                basic operations.
                <p>
                You can open new trees, close views or close all <i>CGNS.NAV</i> from this
                panel. 
                
                <h3>Table entries</h3>
                Each line of the control panel is a view. Each view has a number, which
                counts from 001 up to the max number of views you have. The status and the
                type of the view are indicated by an icon.<i>We use the term <b>view</b>
                when a CGNS.NAV window appears in this list
                and <b>panel</b> when the window doesn't appear.
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
                <tr><td><img source=":/images/icons/save-S-UM.png"></td>
                <td>File can be saved, no modification to save now</td></tr>
                <tr><td><img source=":/images/icons/save-S-M.png"></td>
                <td>File can be saved, some modifications are not saved yet</td></tr>
                <tr><td><img source=":/images/icons/save-US-UM.png"></td>
                <td>File cannot be saved, no modification to save now. You have no write access to the file, 
                you should use <b>save as</b> button.</td></tr>
                <tr><td><img source=":/images/icons/save-US-M.png"></td>
                <td>File cannot be saved, some modifications are not saved yet. You have no write access to the file, 
                you should use <b>save as</b> button.</td></tr>
                </table>
                <p>
                The second column of a control panel line is the view type:
                
                <table>
                <tr><td><img source=":/images/icons/tree-load.png"></td>
                <td> Tree view - Open from control panel</td></tr>
                <tr><td><img source=":/images/icons/vtkview.png"></td>
                <td> VTK view - Open from tree view</td></tr>
                <tr><td><img source=":/images/icons/operate-execute.png"></td>
                <td> Query view - Open from tree view</td></tr>
                <tr><td><img source=":/images/icons/form.png"></td>
                <td> Form view - Open from tree view</td></tr>
                <tr><td><img source=":/images/icons/operate-list.png"></td>
                <td> Selection view - Open from tree view</td></tr>
                <tr><td><img source=":/images/icons/check-all.png"></td>
                <td> Diagnosis view - Open from tree view</td></tr>
                <tr><td><img source=":/images/icons/link.png"></td>
                <td> Link view - Open from tree view</td></tr>
                </table>
                
                <p>
                The remaining columns of a control panel line are the directory and the 
                filename of the CGNS/Python tree and the target node of the tree. 
                These values may be empty if not relevant. For example if you create
                a tree from scratch, the directory and the file name are generated and thus
                would require a <b>save as</b> to set these values.
                
                All <b>CGNS.NAV</b> windows have a <img source=":/images/icons/top.png">
                button that raises the control panel window. The other way, you select
                a view line in the control panel and press <i>[space]</i> to raise the
                corresponding view.
                
                You can close a single view, all views related to one tree or all the trees
                using the right click menu, on a row.
                
                <h3>Buttons</h3>
                <p>
                <table>
                <tr><td><img source=":/images/icons/tree-load-g.png"></td>
                <td> load last used file</td></tr>
                <tr><td><img source=":/images/icons/tree-load.png"></td>
                <td> load a file, open the file dialog window</td></tr>
                <tr><td><img source=":/images/icons/tree-new.png"></td>
                <td> create a new CGNS/Python tree from scratch</td></tr>
                <tr><td><img source=":/images/icons/pattern.png"></td>
                <td> open the pattern database</td></tr>
                <tr><td><img source=":/images/icons/options-view.png"></td>
                <td> open the user options panel</td></tr>
                <tr><td><img source=":/images/icons/help-view.png"></td>
                <td> help</td></tr>
                <tr><td><img source=":/images/icons/view-help.png"></td>
                <td> about CGNS.NAV</td></tr>
                <tr><td><img source=":/images/icons/close-view.png"></td>
                <td> close CGNS.NAV and all its views</td></tr>
                </table>
                
                """,
    ),
    # --------------------------------------------------------------------------
    "Tree": (
        "Tree view",
        """
             <h2>Tree view</h2>
             
             The main window you would use for browsing/editing a CGNS/Python tree is
             the <i>Tree view</i>.
             
             <h3>Hierarchical view</h3>
             The hierarchical view is a classical tree browser. You click just before the node name to 
             expand/collapse its children node. It supports copy/cut/paste as well
             as line editing for node name change, node sidstype change and node value change.
             <p>
             
             Bold names are user defined names, the non-bold are names found in the CGNS/SIDS.
             <p>
             You can use <i>[up][down]</i>arrows to move from a node to another, if you press <i>[control]</i> 
             together with the <i>[up][down]</i> then you move from nodes of a same level.
             Pressing <i>[space]</i> sets/unsets the selection flag on the current node.
             
             <h3>Current node and selected nodes</h3>
             
             The current node is the highlighted line. The selected nodes are the lines
             with a blue flag on. Some commands require a current node to run, for example
             the <i>Form view</i> requires it. Some other commands require a list of
             selected nodes, or example the check, or the <i>paste as child for each selected node</i>. This latter 
             command is in the right-click menu.
             
             <h3>Flags</h3>
             
             There a 4 flag columns: <b>L</b> is for link flag, <b>M</b> is mark flag,
             <b>C</b> is check flag and the last <b>U</b> is user flag.<br><br>
             
             There are two link flags, <img source=":/images/icons/link.png"> indicates
             the root node of a linked-to node. The actual node is in file used for the
             current tree, but the node value and its children are located into another
             file. Use the <img source=":/images/icons/link-view.png"> icon to open the
             link view, that shows all existing links for the current tree. Or you can move
             the mouse pointer on the link icon, the actual linked-to reference would
             be displayed.
             The second link flag is
             <img source=":/images/icons/link-child.png"> that indicates the node has a
             linked-to ancestor node (its direct parent or any other parent). Depending
             on the options you set, you may or may not have the right to change this
             child-link node name, type or value.<br><br>
             
             The mark flag is set when you run a selection, or when you mark it by
             yourself by pressing <i>[space]</i>. The list of all selected nodes, for
             example after running a large query, is displayed using the
             <img source=":/images/icons/operate-list.png"> icon.<br><br>
             
             The check flag is set when you run a check with 
             <img source=":/images/icons/check-all.png">, the diagnosis list is
             open with <img source=":/images/icons/check-list.png">.
             Nodes can have a warning <img source=":/images/icons/check-warn.png">
             or an error <img source=":/images/icons/check-fail.png"> flag. Node without
             flags are ok.<br><br>
             
             The last flag column is for the user flag. The user sets/unsets the flag
             for each node by selecting the node and pressing one of the <i>[0-9]</i> keys.
             Then, by default, the corresponding number is set as flag (for example you
             have a <img source=":/images/icons/user-5.png"> if you press <i>[5]</i>.
             The flag is an informative flag, it is not used by CGNS.NAV functions.<br><br>
             
             The snapshot below shows the flag columns with various flag settings.<br><br>
             <img source=":/images/icons/help-01.png"><br>
             
             
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
             <tr><td><img source=":/images/icons/save.png"></td>
             <td> save file using the last used file name and save parameters</td></tr>
             <tr><td><img source=":/images/icons/tree-save.png"></td>
             <td> save as, as a new directory and/or file name, allows to change save
             parameters such as link management</td></tr>
             <tr><td><img source=":/images/icons/pattern-save.png"></td>
             <td> save as a pattern (not available yet)</td></tr>
             <tr><td><img source=":/images/icons/level-out.png"></td>
             <td> expand one level of children nodes</td></tr>
             <tr><td><img source=":/images/icons/level-all.png"></td>
             <td> switch expand all / unexpand all</td></tr>
             <tr><td><img source=":/images/icons/level-in.png"></td>
             <td> unexpand one level of children nodes</td></tr>
             <tr><td><img source=":/images/icons/flag-all.png"></td>
             <td> set the selection flag on all the nodes. If a node is not visible, then it is flagged as well and 
             all subsequent operation on selection list would take this invisible node as argument. Note that such an 
             invisble node is actually in the CGNS/Python tree but it is not displayed</td></tr>
             <tr><td><img source=":/images/icons/flag-revert.png"></td>
             <td> revert nodes selection flag</td></tr>
             <tr><td><img source=":/images/icons/flag-none.png"></td>
             <td> remove all selection flags</td></tr>
             <tr><td><img source=":/images/icons/operate-list.png"></td>
             <td> open the selection view, gives the list of all selected nodes</td></tr>
             <tr><td><img source=":/images/icons/check-all.png"></td>
             <td> check nodes using <b>CGNS.VAL</b> tool. If the selection list is not
             empty, then only the selected nodes are checked. If the selection list is
             empty, all nodes of the tree are checked.</td></tr>
             <tr><td><img source=":/images/icons/check-clear.png"></td>
             <td> remove all check flags</td></tr>
             <tr><td><img source=":/images/icons/check-list.png"></td>
             <td> open the diagnosis view which details the check log</td></tr>
             <tr><td><img source=":/images/icons/link-select.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/link-add.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/link-delete.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/form-open.png"></td>
             <td> open the form view, gives details on a node including its data</td></tr>
             <tr><td><img source=":/images/icons/vtk.png"></td>
             <td> open the VTK view, a 3D view on the actual mesh and its associated data.
             This requires a correct parsing of CGNS/Python tree, if the tool is not able
             to understand the data (for example if your tree is not CGNS/SIDS compliant,
             then no window is open.</td></tr>
             <tr><td><img source=":/images/icons/plot.png"></td>
             <td> open the Plot view, a 2D view on some tree data.
             Same remarks as VTK view.</td></tr>
             <tr><td><img source=":/images/icons/pattern-view.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/link-view.png"></td>
             <td> open the Link view, gives details on the links</td></tr>
             <tr><td><img source=":/images/icons/operate-view.png"></td>
             <td> open the Query view, a powerful mean to edit and run Python commands on the tree</td></tr>
             <tr><td><img source=":/images/icons/check-view.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/tools-view.png"></td>
             <td> not available yet</td></tr>
             <tr><td><img source=":/images/icons/snapshot.png"></td>
             <td> creates a bitmap file with the snapshot of the current tree view.
             See user options for directory and file used for this snapshot.</td></tr>
             </table>
             
             """,
    ),
    # --------------------------------------------------------------------------
    "VTK": (
        "VTK view",
        """
            <h2>VTK view</h2>
            
            <h3>Selection list</h3>
            <p>
            <table>
            <tr><td><img source=":/images/icons/selected.png"></td>
            <td>The selection list entry is selected for operations</td></tr>
            <tr><td><img source=":/images/icons/unselected.png"></td>
            <td>The selection list entry is NOT selected for operations</td></tr>
            <tr><td><img source=":/images/icons/hidden.png"></td>
            <td>The selection list entry is hidden, the data is still there but it is not
            displayed in the window.</td></tr>
            </table>
            
            <h3>Icons and buttons</h3>
            <p>
            <table>
            <tr><td><img source=":/images/icons/zoompoint.png"></td>
            <td>Switch to the zoom-window mode. You draw a window using the mouse left
            button and the zoom would fit to the defined window.</td></tr>
            <tr><td><img source=":/images/icons/zoom-actor.png"></td>
            <td>Reset the view to fit the object in the window</td></tr>
            <tr><td><img source=":/images/icons/lock-legend.png"></td>
            <td>Switch to move legend mode, used to mose the reference axis draw, the
            legend and other extra information displayed on the view.</td></tr>
            <tr><td><img source=":/images/icons/value.png"></td>
            <td>Switch to the show value mode, in the case you have actual values,
            the pick of a node would show corresponding value.</td></tr>
            <tr><td><img source=":/images/icons/colors.png"></td>
            <td>Randomly change the colors of view objects.</td></tr>
            <tr><td><img source=":/images/icons/colors-bw.png"></td>
            <td>Switch the view background to balck/white.</td></tr>
            </table>
            
            
            <h3>Displaying actual values</h3>
            <p>
            
            
            <h3>User defined camera views</h3>
            <p>
            You can the current camera and reset to this stored point of view later on.
            <table>
            <tr><td><img source=":/images/icons/camera-snap.png"></td>
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
            Keys are not case-sensitive letters, <b>s</b> and <b>S</b> are the same.
            
            <b>s</b> Surface mode rendering (toggle with wire)<br>
            <b>w</b> Wire mode rendering (toggle with surface)<br>
            <b>q</b> Surface and wire mode rendering (toggle with wire)<br>
            <b>r</b> Reset view to fit to full object<br>
            
            <b>[space]</b> Throw a ray on the view and select all actors cuting the ray<br>
            <b>z</b> Same as [space]<br>
            <b>u</b> Unselect all<br>
            <b>p</b> Pick a point in the selected Zone<br>
            
            <h4>Tips</h4>
            
            To unselect all and clear the list of selected actors, just move the mouse on
            the graphic window background and press [space].
            """,
    ),
    # --------------------------------------------------------------------------
    "Link": (
        "Link view",
        """
             <h2>Link view</h2>
             
             <b>When you copy a node path you SHOULD NOT add the CGNSTree root</b>
             
             <h3>Buttons</h3>
             <p>
             
             """,
    ),
    # --------------------------------------------------------------------------
    "Form": (
        "Form view",
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
             
             """,
    ),
    # --------------------------------------------------------------------------
    "Query": (
        "Query view",
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
              <tr><td><b>LINKS</b></td><td>the list of links info</td></tr>
              <tr><td><b>SKIPS</b></td><td>the list of skips info</td></tr>
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
              <tr><td><img source=":/images/icons/operate-add.png"></td>
              <td>Add a new query</td></tr>
              <tr><td><img source=":/images/icons/operate-delete.png"></td>
              <td>Remove an existing query</td></tr>
              <tr><td><img source=":/images/icons/operate-save.png"></td>
              <td>Save all queries</td></tr>
              </table>
              
              """,
    ),
    # --------------------------------------------------------------------------
    "Option": (
        "Option panel",
        """
               <h2>Option panel</h2>
               <font color=red>Oups! the Option panel is a MODAL window, which means
               it blocks all other windows until you answer to it. So if you want to
               browse this help you have to close the Option panel window first!</font>
               <h3>Buttons</h3>
               
               <h3>Message codes</h3>
               <table>
               <tr><td>100</td><td> * </td><td>Version and Copyright notice</td></tr>
               <tr><td>101</td><td> - </td><td>CGNS.NAV Exit double check</td></tr>
               <tr><td>110</td><td> - </td><td>Directory not found during file open</td></tr>
               <tr><td>120</td><td> - </td><td>Clear file history - current not found file</td></tr>
               <tr><td>121</td><td> - </td><td>Clear file history - all not found files</td></tr>
               <tr><td>122</td><td> - </td><td>Clear file history - all files</td></tr>
               <tr><td>130</td><td> - </td><td>Save abort</td></tr>
               <tr><td>131</td><td> - </td><td>Save abort (execption caught)</td></tr>
               <tr><td>132</td><td> - </td><td>Save warning about file overwriting issue</td></tr>
               <tr><td>200</td><td> * </td><td>Error during load (returns actual CGNS/HDF5 layer error)</td></tr>
               <tr><td>201</td><td> * </td><td>Fatal error during load</td></tr>
               <tr><td>500</td><td> - </td><td>Bad data for a VTK display</td></tr>
               <tr><td>501</td><td> - </td><td>VTK version lower than v6.0</td></tr>
               <tr><td>502</td><td> - </td><td>VTK version lower than v5.8</td></tr>
               <tr><td>302</td><td> - </td><td>Node selection required for Form view</td></tr>
               <tr><td>310</td><td> - </td><td>Create link as new node</td></tr>
               <tr><td>311</td><td> - </td><td>Remove link entry</td></tr>
               <tr><td>370</td><td> - </td><td>Leave query panel</td></tr>
               <tr><td>371</td><td> - </td><td>Apply changes to all</td></tr>
               <tr><td>400</td><td> * </td><td>Show diag view grammer used</td></tr>
               </table>
               
               <p>
               <img source=":/images/icons/unselected.png">
               """,
    ),
    # --------------------------------------------------------------------------
    "File": (
        "File panel",
        """
             <h2>File panel</h2>
             <font color=red>Oups! the File panel is a MODAL window, which means
             it blocks all other windows until you answer to it. So if you want to
             browse this help you have to close the File panel window first!</font>
             <p>
             The same panel is used for both <b>load</b> and <b>save</b> file selection.
             The <b>Cancel</b> button would abort the current load or save
             operation, but once the the load/save process is strated there is no way
             to abort it.
             <p>
             There are two tabs, the <i>Selection</i> tab is the default one, you select
             there the directory and filename of the target file you want to load or to
             save. The <I>Options</i> tab sets the behavior of the file filter and the
             history of your loads/saves.
             <P>
             The <i>File panel</i> filters the CGNS/HDF/ADF files depending on the options
             you set. Files can be sorted by name, size, type or modification date by
             clicking on the corresponding column header.
             <p>
             Unchecking the <i>Show directories</i> checkbox would mask all directories
             from the table, thus making it more easy to find your file in large lists.
             
             <h3>Load</h3>
             The <i>File panel</i> options checkboxes, such as <i>Do Not load large data</i>
             are used only for the current load/save. The initial values of these checkboxes,
             each time you open the panel, are get from your overall options (see the
             <img source=":/images/icons/option-view.png"> <i>Options Panel</i>).
             <p>
             <img source=":/images/icons/h_files_checks.png">
             <table>
             <tr><td><b>Do not load large data</b></td><td>Skip large arrays, the threshold
             is set in the <i>Options Panel</i></td></tr>
             <tr><td><b>Follow links</b></td><td>Recursively load files you can reach through linksSkip large arrays, 
             the threshold is
             set in the <i>Options Panel</i></td></tr>
             <tr><td><b>Open as read-only</b></td><td>Prevent any attempt to modify the
             tree. This guard is unlocked if you do a <I>Save as</i>, the tree is then
             associated to a new file and becames writable.</td></tr>
             </table>
             
             <h3>Save</h3>
             The save policy is complex as saving a CGNS tree can be an update or
             an overwrite. 
             The save checkboxes are <i>overwrite</i> and <i>delete missing</i>, the behavior
             of the same is summerized in this table:
             <table>
             <tr><td><img source=":/images/icons/w_file_checks_00.png"></td><td></td></tr>
             <tr><td><img source=":/images/icons/w_file_checks_10.png"></td><td></td></tr>
             <tr><td><img source=":/images/icons/w_file_checks_01.png"></td><td></td></tr>
             <tr><td><img source=":/images/icons/w_file_checks_11.png"></td><td></td></tr>
             </table>
             
             <h3>Buttons</h3>
             <p>
             <img source=":/images/icons/unselected.png">
             """,
    ),
    # --------------------------------------------------------------------------
    "Selection": (
        "Selection view",
        """
                  <h2>Selection view</h2>
                  
                  The <i>Selection view</i> lists the selected nodes resulting from a query or
                  user performed selection. The only action you can perform on this list is
                  to change the values, datatypes or SIDS types, or to save the list.
                  
                  The actual purpose of this view is to reduce the view of a tree for local
                  modifications. For example you perform a query to retrieve a specific node
                  and then you edit the selected nodes.
                  
                  You can run the query and display the result at once using the command line,
                  for example:
                  <pre>CGNS.NAV -q '100. Family boundaries' File.cgns</pre>
                  
                  <h3>Buttons</h3>
                  <p>
                  <table>
                  <tr><td><img source=":/images/icons/control.png">
                  <img source=":/images/icons/unselected.png">
                  <img source=":/images/icons/node-sids-closed.png">
                  </td>
                  <td> unsused</td></tr>
                  <tr><td><img source=":/images/icons/select-add.png"></td>
                  <td> select all lines</td></tr>
                  <tr><td><img source=":/images/icons/falg-revert.png"></td>
                  <td> revert line selection</td></tr>
                  <tr><td><img source=":/images/icons/select-delete.png"></td>
                  <td> unselect all lines</td></tr>
                  <tr><td><img source=":/images/icons/flag-none"></td>
                  <td> remove a line from the selection line</td></tr>
                  <tr><td><img source=":/images/icons/select-save"></td>
                  <td> save the selection list in a file</td></tr>
                  </table>
                  
                  <h3>User defined edition</h3>
                  The node value editing can be a plain text edition or a combo-box enumerate.
                  In that latter case the user can define a function to map the list of
                  allowed value with respect to the node path (name for example) and the node
                  type path (that is the list of types for each node in the path to the
                  current node). The function should be a python class in a python file,
                  the file name is <pre>default.py</pre> in the directory
                  <pre>$HOME/.CGNS.NAV/funtions</pre>. There is an example of such a file:
                  <pre>
                  # -----------------------------------------------------------------
                  # Mandatory name
                  #
                  import CGNS.PAT.cgnskeywords as CGK
                  
                  class Q7UserFunction(object):
                    __attributes={
                      'artviscosity':['none','dissca','dismat','dismrt'],
                      'fluid':['rg','pg'],
                      'flux':['jameson','roe','vleer','coquel_d','coquel_i','ausmp','rbc','rbci'],
                      'ode':['backwardeuler','rk4'],
                      'phymod':['euler','nslam','nstur'],
                    }
                    # --- mandatory method
                    def getEnumerate(self,namepath,typepath):
                      if (namepath[-1] in self.__attributes):
                          return self.__attributes[namepath[-1]]
                      else:
                          return CGK.cgnsenums[typepath[-1]]
                      return None
                  
                  # --- last line
                  </pre>
                  """,
    ),
    # --------------------------------------------------------------------------
    "Diagnosis": (
        "Diagnosis view",
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
                  <tr><td><img source=":/images/icons/level-in.png"></td>
                  <td> expand all errors/warnings</td></tr>
                  <tr><td><img source=":/images/icons/level-out.png"></td>
                  <td> collapse all errors/warnings</td></tr>
                  <tr><td><img source=":/images/icons/node-sids-opened.png"></td>
                  <td> go to previous filtered error/warning</td></tr>
                  <tr><td><img source=":/images/icons/selected.png"></td>
                  <td> go to next filtered error/warning</td></tr>
                  <tr><td><img source=":/images/icons/check-save.png"></td>
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
                  
                  """,
    ),
    # --------------------------------------------------------------------------
    "Info": (
        "Info panel",
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
             """,
    ),
    # --------------------------------------------------------------------------
    "Pattern": (
        "Pattern panel",
        """
                <h2>Pattern panel</h2>
                
                """,
    ),
    # --------------------------------------------------------------------------
    "Tools": (
        "Tools panel",
        """
              <h2>Tools panel</h2>
              
              <h1>Search</h1>
              The SIDS type can be selected as a choice in the CGNS/SIDS types or you
              can enter your own pattern for regexp search. In that later case, you have to
              press [ENTER] once you have typed down the actual search string.
              
              The Children checks are performed with a OR clause.
              If you require a child to be named 'FamilyName' for example, then the check
              is True if at least one child has this name. With the NOT clause, the check
              is True if NO child has the name 'FamilyName'.
              
              """,
    ),
    # --------------------------------------------------------------------------
    "Message": (
        "Message window",
        """
                <h2>About Messages</h2>
                
                Some info messages have the <i>don't show again</b> check box.
                If you set this once, you would never see the message again, unless you
                re-activate it in the <b>Option Panel</b>
                
                """,
    ),
    # --------------------------------------------------------------------------
}

# --- last line
