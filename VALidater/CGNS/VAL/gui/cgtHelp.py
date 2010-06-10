# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCCCCC - $Id: cgtHelp.py 22 2005-02-02 09:57:08Z  $
#
# See file COPYING in the root directory of this Python module source 
# tree for license information. 
#
#
#
from CCCCC.gui import __cgtvid__
helpAbout="""
CGT - CGNS Tree Viewer - v%s
"""%__cgtvid__

helpKeys="""
In the main view window, you can click on the label itself or on
the [+] [-] little icon at the branches jonctions.
A red box highlights the selected node.

1 -------------------------------------------

A mouse right click on this [+] [-] icon toggles the open/close
of the sub-tree starting from this node. Other clicks on this
icons have no effect.

If the 'Expand all sub-tree' flag is active (see Display menu), the
whole sub-tree below the current node is shown.

Display of the DataArray contents depends on the data flag (see Display menu).

You can display node Name/Type/Attributes, check flags in Display menu.

2 -------------------------------------------

The table below gives the click table when your mouse is on the
node label:

<Click-Right>       Select node
<Click-Middle>      Node menu
<Click-Left>        Node full description

<CTRL-Click-Middle> Mark node (appears in red)

3 -------------------------------------------

You can also use the keyboard to navigate:

<Down>          Next node
<Up>            Previous node
<Left>          Descend in tree (open node)
<Right>         Climb the tree

<Home>          Top node
<End>           Bottom node

<Space>         Toggle node (open/close)
<Control-Space> Toggle level (open/close all nodes at this level)

---------------------------------------------
"""
helpMenus="""
Top window menus:

Node menus:
"""
helpCGNS="""
CFD General Notation System - http://www.cgns.org

The [Computational Fluid Dynamics] general notation system
defines a grammar for CFD data structure representation.

Please refer to the web site to get information on CGNS.

"""
