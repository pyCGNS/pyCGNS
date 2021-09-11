#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
colordict = {"high": "#7ebaff", "low": "#78b5e9", "high2": "#bad5f4", "low2": "#bad5ff"}

Q7CONTROLVIEWSTYLESHEET = (
    """
 QTableView {
 show-decoration-selected: 1;
 }

 QTableView::item:selected {
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 %(high)s,stop:1 %(low)s);
 color:black;
 border: 1px solid #bfcde4;
 }

 QTableView::item
 {
 border: 0px;
 padding: 0px;
 }
"""
    % colordict
)

Q7TABLEVIEWSTYLESHEET = (
    """
 Q7TableView {
 show-decoration-selected: 1;
 }

 Q7TableView::item:selected
 {
  border: 1px solid %(high)s;
  font: bold;
 }

 Q7TableView::item
 {
  border: 0px;
  padding: 0px;
  font: fixed 8;
 }
"""
    % colordict
)

Q7TREEVIEWSTYLESHEET = (
    """
 QTreeView {
 show-decoration-selected: 1;
 }

 QTreeView::branch {
 background: palette(base);
 }

 QTreeView::item [diffNA="true"] { background-color: yellow }

 QTreeView::item {
 border: 1px solid #d9d9d9;
 border-top-color: transparent;
 border-bottom-color: transparent;
 text-align:right;
 color:black;
 }

 QTreeView::item:selected {
 border: 1px solid #567dbc;
 }

 QTreeView::item:selected:active{
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 %(high)s,stop:1 %(low)s);
 }

 QTreeView::item:selected:!active {
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 %(high)s,stop:1 %(low)s);
 }

 QTreeView::branch:has-siblings:!adjoins-item {
 border-image: url(:/images/icons/vline.png) 0;
 }

 QTreeView::branch:has-siblings:adjoins-item {
 border-image: url(:/images/icons/branch-more.png) 0;
 }

 QTreeView::branch:!has-children:!has-siblings:adjoins-item {
 border-image: url(:/images/icons/branch-end.png) 0;
 }

 QTreeView::branch:closed:has-children:has-siblings {
 border-image: url(:/images/icons/branch-more-p.png) 0;
 }

 QTreeView::branch:has-children:!has-siblings:closed {
 border-image: url(:/images/icons/branch-end-p.png) 0;
 }

 QTreeView::branch:open:has-children:has-siblings {
 background: magenta;
 }

 QTreeView::branch:open:has-children:!has-siblings {
 border-image: url(:/images/icons/branch-end.png) 0;
 }

"""
    % colordict
)

# -----------------------------------------------------------------
