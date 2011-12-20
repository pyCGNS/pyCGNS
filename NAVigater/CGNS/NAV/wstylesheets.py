#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------

Q7TREEVIEWSTYLESHEET="""
 QTreeView {
 show-decoration-selected: 1;
 }

QTreeView::branch {
background: palette(base);
 }

 QTreeView::item {
 border: 1px solid #d9d9d9;
 border-top-color: transparent;
 border-bottom-color: transparent;
 text-align:right;
 }

 QTreeView::item:hover {
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e7effd,stop:1 #cbdaf1);
 border: 1px solid #bfcde4;
 }

 QTreeView::item:selected {
 border: 1px solid #567dbc;
 }

 QTreeView::item:selected:active{
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6ea1f1,stop:1 #567dbc);
 }

 QTreeView::item:selected:!active {
 background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6b9be8,stop:1 #577fbf);
 }

 QTreeView::branch:has-siblings:!adjoins-item {
 border-image: url(:/images/icons/vline.gif) 0;
 }

 QTreeView::branch:has-siblings:adjoins-item {
 border-image: url(:/images/icons/branch-more.gif) 0;
 }

 QTreeView::branch:!has-children:!has-siblings:adjoins-item {
 border-image: url(:/images/icons/branch-end.gif) 0;
 }

 QTreeView::branch:closed:has-children:has-siblings {
 border-image: url(:/images/icons/branch-more-p.gif) 0;
 }

 QTreeView::branch:has-children:!has-siblings:closed {
 border-image: url(:/images/icons/branch-end-p.gif) 0;
 }

 QTreeView::branch:open:has-children:has-siblings {
 background: magenta;
 }

 QTreeView::branch:open:has-children:!has-siblings {
 border-image: url(:/images/icons/branch-end.gif) 0;
 }
"""

# -----------------------------------------------------------------

