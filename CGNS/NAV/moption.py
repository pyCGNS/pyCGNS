# -*- coding: utf-8 -*-
#
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import shutil
import os
import os.path as op
import sys
import imp
import tempfile

from time import gmtime, strftime
from CGNS.pyCGNSconfig import version as __vid__
from CGNS.pyCGNSconfig import HAS_MSW

WINOPTS = HAS_MSW

from ..PAT import cgnskeywords as CGK

from qtpy.QtGui import QFont, QFontDatabase, QTextCursor

# User file paths:
conf_path = op.join(op.expanduser("~"), ".CGNS.NAV")
tmp_path = "/tmp"
if sys.platform == "win32":
    tmp_path = os.environ["TEMP"]


# -----------------------------------------------------------------
def removeSubDirAndFiles(path):
    shutil.rmtree(path)


def copyOneFile(src, dst):
    f1 = open(src, "r")
    l1 = f1.readlines()
    f1.close()
    if op.exists(dst):
        os.remove(dst)
    f2 = open(dst, "w")
    f2.writelines(l1)
    f2.close()


# -----------------------------------------------------------------
class Q7OptionContext(object):
    Q_VAR_NODE = "NODE"
    Q_VAR_PARENT = "PARENT"
    Q_VAR_NAME = "NAME"
    Q_VAR_VALUE = "VALUE"
    Q_VAR_CGNSTYPE = "SIDSTYPE"
    Q_VAR_CHILDREN = "CHILDREN"
    Q_VAR_TREE = "TREE"
    Q_VAR_PATH = "PATH"
    Q_VAR_LINKS = "LINKS"
    Q_VAR_SKIPS = "SKIPS"
    Q_VAR_RESULT = "RESULT"
    Q_VAR_USER = "ARGS"
    Q_VAR_SELECTED = "SELECTED"
    Q_VAR_RESULT_LIST = "__Q7_QUERY_RESULT__"

    Q_SCRIPT_PRE = """
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnslib as CGL
import numpy
"""

    Q_FILE_PRE = """
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnslib as CGL
import CGNS.NAV.moption as CGO
import numpy
"""

    Q_SCRIPT_POST = """
try:
  %s[0]=%s
except NameError:
  %s[0]=None
""" % (
        Q_VAR_RESULT_LIST,
        Q_VAR_RESULT,
        Q_VAR_RESULT_LIST,
    )

    _depends = {
        "MaxDisplayDataSize": ["DoNotDisplayLargeData"],
        "MaxLoadDataSize": ["DoNotLoadLargeArrays"],
        "MaxRecursionLevel": ["RecursiveTreeDisplay"],
    }
    _HasProPackage = True
    CHLoneTrace = False
    QueryNoException = False
    ActivateMultiThreading = False
    NAVTrace = False
    AutoExpand = False
    RecursiveTreeDisplay = False
    OneViewPerTreeNode = False
    ShowTableIndex = True
    RecursiveSIDSPatternsLoad = True
    DoNotDisplayLargeData = False
    CheckOnTheFly = False
    FollowLinksAtLoad = True
    DoNotFollowLinksAtSave = True
    AddCurrentDirInSearch = True
    AddRootDirInSearch = True
    DoNotLoadLargeArrays = True
    ShowSIDSStatusColumn = True
    ForceSIDSLegacyMapping = False
    ForceFortranFlag = True
    FilterCGNSFiles = True
    FilterHDFFiles = True
    FilterOwnFiles = True
    FileUpdateRemovesChildren = True
    TransposeArrayForView = True
    Show1DAsPlain = True
    ShowSIDSColumn = True
    ShowLinkColumn = True
    ShowSelectColumn = True
    ShowUserColumn = True
    ShowCheckColumn = True
    ShowShapeColumn = True
    ShowDataTypeColumn = True
    UserCSS = ""
    SelectionListDirectory = op.normpath(op.join(conf_path, "selections"))
    QueriesDirectory = op.normpath(op.join(conf_path, "queries"))
    FunctionsDirectory = op.normpath(op.join(conf_path, "functions"))
    SnapShotDirectory = op.normpath(op.join(conf_path, "snapshots"))
    _HistoryFileName = op.normpath(op.join(conf_path, "historyfile.py"))
    _OptionsFileName = op.normpath(op.join(conf_path, "optionsfile.py"))
    _QueriesDefaultFile = "default.py"
    _FunctionsDefaultFile = "default.py"
    IgnoredMessages = []
    LinkSearchPathList = []
    ProfileSearchPathList = []
    GrammarSearchPathList = []
    ValKeyList = ["sample"]
    CGNSFileExtension = [".cgns", ".adf"]
    HDFFileExtension = [".hdf", ".hdf5"]
    OwnFileExtension = [".cgh"]
    MaxLoadDataSize = 1000
    MaxDisplayDataSize = 1000
    MaxRecursionLevel = 7
    ADFConversionCom = "cgnsconvert"
    TemporaryDirectory = op.normpath(tmp_path)
    _ConvertADFFiles = True
    _ToolName = "CGNS.NAV"
    _ToolVersion = "%s" % __vid__

    _CopyrightNotice = """
Copyright (c) Marc Poinot <br>
Copyright (c) Safran - www.safran-group.com<br><br>
Copyright (c) Onera - www.onera.fr<br><br>
<b>all other copyrights and used versions listed below</b>

<hr>
<h3>Contributors (alphabetic order)</h3>
<table>
<tr><td>Florent Cayré</td><td>-SNECMA, France</td></tr>
<tr><td>Alexandre Fayolle</td><td>-LOGILAB, France</td></tr>
<tr><td>Xavier Garnaud</td><td>-SAFRAN, France</td></tr>
<tr><td>Loic Hauchard</td><td>-ONERA (Student INSA Rouen, France)</td></tr>
<tr><td>Elise Hénaux</td><td>-ONERA (Student FIIFO Orsay, France)</td></tr>
<tr><td>Grégory Laheurte</td><td>-ONERA, France</td></tr>
<tr><td>Pierre-Jacques Legay</td><td>-BERTIN, France</td></tr>
<tr><td>Bill Perkins</td><td>-Pacific Northwest Ntl Lab, U.S.A.</td></tr>
<tr><td>Mickael Philit</td><td>-SAFRAN, France</td></tr>
<tr><td>Jérôme Regis</td><td>-STILOG, France</td></tr>
<tr><td>Benoit Rodriguez</td><td>-ONERA, France</td></tr>
<tr><td>Tristan Soubrié</td><td>-ANDHEO, France</td></tr>
<tr><td>Francois Thirifays</td><td>-CENAERO, Belgique</td></tr>
<tr><td>Simon Verley</td><td>-ONERA, France</td></tr>
<tr><td>Ching-Yang Wang</td><td>-U.S.A.</td></tr>
</table>

<h2>Copyrights & Versions</h2>
<hr>
%(pycgnsversion)s<br>
All <b>pyCGNS</b> 
rights reserved in accordance with GPL v2 <br><br>
<b>NO WARRANTY :</b><br>
Check GPL v2 sections 15 and 16
about <font color=red>loss of data or corrupted data</font><br>
<hr>
%(pyqtversion)s<br>
PyQt Copyright (c) Riverbank Computing Limited. <br>
Usage within the terms of the GPL v2.<br>
All Rights Reserved.<br>
<hr>
%(qtversion)s<br>
Qt Copyright (c)<br>
The Qt4 Library is (c) 2011 Nokia Corporation and/or its subsidiary(-ies),
and is licensed under the GNU Lesser General Public License version 2.1
with Nokia Qt LGPL exception version 1.1. <br>
<hr>
%(pythonversion)s<br>
Python Copyright (c)<br>
Copyright (c) 2001-2011 Python Software Foundation.<br>
All Rights Reserved.<br>
<br>
Copyright (c) 2000 BeOpen.com.<br>
All Rights Reserved.<br>
<br>
Copyright (c) 1995-2001 Corporation for National Research Initiatives.<br>
All Rights Reserved.<br>
<br>
Copyright (c) 1991-1995 Stichting Mathematisch Centrum, Amsterdam.<br>
All Rights Reserved.<br>
<br>
<hr>
%(numpyversion)s<br>
Numpy Copyright (c)<br>
Copyright (c) 2005, NumPy Developers<br>
<hr>
%(hdf5version)s<br>
HDF5 Copyright (c)<br>
HDF5 (Hierarchical Data Format 5) Software Library and Utilities<br>
Copyright 2006-2013 by The HDF Group.<br>
NCSA HDF5 (Hierarchical Data Format 5) Software Library and Utilities<br>
Copyright 1998-2006 by the Board of Trustees of the University of Illinois.<br>
<hr>
%(vtkversion)s<br>
VTK Copyright (c)<br>
Copyright (c) 1993-2008 Ken Martin, Will Schroeder, Bill Lorensen <br>
All rights reserved.<br>
<hr>
%(cythonversion)s<br>
Cython Copyright (c)<br>
(c) Copyright 2012, Stefan Behnel, Robert Bradshaw, Dag Sverre Seljebotn,
Greg Ewing, William Stein, Gabriel Gellner, et al.
<hr>
Icons Copyright (c)<br>
All these nice icons are provided or are modified versions of original
icons provided by Mark James (Birmingham, UK).<br>
Please visit his web site: http://www.famfamfam.com/<br>

"""
    # __fv = QFont(QFontDatabase().families()[0]) # Spyder-like but needs a QApplication
    __fv = QFont("Courier new")
    __fc = QFont("Courier new")

    _Label_Font = QFont(__fv)
    _Button_Font = QFont(__fv)
    _Table_Font = QFont(__fc)
    _Edit_Font = QFont(__fc)
    _RName_Font = QFont(__fc)
    _NName_Font = QFont(__fc)
    _NName_Font.setBold(True)

    _Default_Fonts = {
        "Label_Family": _Label_Font.family(),
        "Label_Size": _Label_Font.pointSize(),
        "Label_Bold": False,
        "Label_Italic": False,
        "Table_Family": _Table_Font.family(),
        "Table_Size": _Table_Font.pointSize(),
        "Table_Bold": False,
        "Table_Italic": False,
        "Edit_Family": _Edit_Font.family(),
        "Edit_Size": _Edit_Font.pointSize(),
        "Edit_Bold": False,
        "Edit_Italic": False,
        "Button_Family": _Button_Font.family(),
        "Button_Size": _Button_Font.pointSize(),
        "Button_Bold": False,
        "Button_Italic": False,
        "RName_Family": _RName_Font.family(),
        "RName_Size": _RName_Font.pointSize(),
        "RName_Bold": False,
        "RName_Italic": False,
        "NName_Family": _NName_Font.family(),
        "NName_Size": _NName_Font.pointSize(),
        "NName_Bold": False,
        "NName_Italic": True,
    }

    UserColors = [
        "gray",
        "red",
        "green",
        "blue",
        "orange",
        None,
        None,
        None,
        None,
        None,
    ]

    _ColorList = {
        "cold_grey": (0.5000, 0.5400, 0.5300),
        "dim_grey": (0.4118, 0.4118, 0.4118),
        "grey": (0.7529, 0.7529, 0.7529),
        "light_grey": (0.8275, 0.8275, 0.8275),
        "slate_grey": (0.4392, 0.5020, 0.5647),
        "slate_grey_dark": (0.1843, 0.3098, 0.3098),
        "slate_grey_light": (0.4667, 0.5333, 0.6000),
        "warm_grey": (0.5000, 0.5000, 0.4100),
        "black": (0.0000, 0.0000, 0.0000),
        "ivory_black": (0.1600, 0.1400, 0.1300),
        "lamp_black": (0.1800, 0.2800, 0.2300),
        "alizarin_crimson": (0.8900, 0.1500, 0.2100),
        "brick": (0.6100, 0.4000, 0.1200),
        "coral": (1.0000, 0.4980, 0.3137),
        "coral_light": (0.9412, 0.5020, 0.5020),
        "deep_pink": (1.0000, 0.0784, 0.5765),
        "firebrick": (0.6980, 0.1333, 0.1333),
        "geranium_lake": (0.8900, 0.0700, 0.1900),
        "hot_pink": (1.0000, 0.4118, 0.7059),
        "light_salmon": (1.0000, 0.6275, 0.4784),
        "madder_lake_deep": (0.8900, 0.1800, 0.1900),
        "maroon": (0.6902, 0.1882, 0.3765),
        "pink": (1.0000, 0.7529, 0.7961),
        "pink_light": (1.0000, 0.7137, 0.7569),
        "raspberry": (0.5300, 0.1500, 0.3400),
        "rose_madder": (0.8900, 0.2100, 0.2200),
        "salmon": (0.9804, 0.5020, 0.4471),
        "tomato": (1.0000, 0.3882, 0.2784),
        "beige": (0.6400, 0.5800, 0.5000),
        "brown": (0.5000, 0.1647, 0.1647),
        "brown_madder": (0.8600, 0.1600, 0.1600),
        "brown_ochre": (0.5300, 0.2600, 0.1200),
        "burlywood": (0.8706, 0.7216, 0.5294),
        "burnt_sienna": (0.5400, 0.2100, 0.0600),
        "burnt_umber": (0.5400, 0.2000, 0.1400),
        "chocolate": (0.8235, 0.4118, 0.1176),
        "deep_ochre": (0.4500, 0.2400, 0.1000),
        "flesh": (1.0000, 0.4900, 0.2500),
        "flesh_ochre": (1.0000, 0.3400, 0.1300),
        "gold_ochre": (0.7800, 0.4700, 0.1500),
        "greenish_umber": (1.0000, 0.2400, 0.0500),
        "khaki": (0.9412, 0.9020, 0.5490),
        "khaki_dark": (0.7412, 0.7176, 0.4196),
        "light_beige": (0.9608, 0.9608, 0.8627),
        "peru": (0.8039, 0.5216, 0.2471),
        "rosy_brown": (0.7373, 0.5608, 0.5608),
        "raw_sienna": (0.7800, 0.3800, 0.0800),
        "raw_umber": (0.4500, 0.2900, 0.0700),
        "sepia": (0.3700, 0.1500, 0.0700),
        "sienna": (0.6275, 0.3216, 0.1765),
        "saddle_brown": (0.5451, 0.2706, 0.0745),
        "sandy_brown": (0.9569, 0.6431, 0.3765),
        "tan": (0.8235, 0.7059, 0.5490),
        "van_dyke_brown": (0.3700, 0.1500, 0.0200),
        "cadmium_orange": (1.0000, 0.3800, 0.0100),
        "carrot": (0.9300, 0.5700, 0.1300),
        "dark_orange": (1.0000, 0.5490, 0.0000),
        "mars_orange": (0.5900, 0.2700, 0.0800),
        "mars_yellow": (0.8900, 0.4400, 0.1000),
        "orange": (1.0000, 0.5000, 0.0000),
        "orange_red": (1.0000, 0.2706, 0.0000),
        "yellow_ochre": (0.8900, 0.5100, 0.0900),
        "aureoline_yellow": (1.0000, 0.6600, 0.1400),
        "banana": (0.8900, 0.8100, 0.3400),
        "cadmium_lemon": (1.0000, 0.8900, 0.0100),
        "cadmium_yellow": (1.0000, 0.6000, 0.0700),
        "cadmium_yellow_light": (1.0000, 0.6900, 0.0600),
        "gold": (1.0000, 0.8431, 0.0000),
        "goldenrod": (0.8549, 0.6471, 0.1255),
        "goldenrod_dark": (0.7216, 0.5255, 0.0431),
        "goldenrod_light": (0.9804, 0.9804, 0.8235),
        "goldenrod_pale": (0.9333, 0.9098, 0.6667),
        "light_goldenrod": (0.9333, 0.8667, 0.5098),
        "melon": (0.8900, 0.6600, 0.4100),
        "naples_yellow_deep": (1.0000, 0.6600, 0.0700),
        "yellow": (1.0000, 1.0000, 0.0000),
        "yellow_light": (1.0000, 1.0000, 0.8784),
        "chartreuse": (0.4980, 1.0000, 0.0000),
        "chrome_oxide_green": (0.4000, 0.5000, 0.0800),
        "cinnabar_green": (0.3800, 0.7000, 0.1600),
        "cobalt_green": (0.2400, 0.5700, 0.2500),
        "emerald_green": (0.0000, 0.7900, 0.3400),
        "forest_green": (0.1333, 0.5451, 0.1333),
        "green": (0.0000, 1.0000, 0.0000),
        "green_dark": (0.0000, 0.3922, 0.0000),
        "green_pale": (0.5961, 0.9843, 0.5961),
        "green_yellow": (0.6784, 1.0000, 0.1843),
        "lawn_green": (0.4863, 0.9882, 0.0000),
        "lime_green": (0.1961, 0.8039, 0.1961),
        "mint": (0.7400, 0.9900, 0.7900),
        "olive": (0.2300, 0.3700, 0.1700),
        "olive_drab": (0.4196, 0.5569, 0.1373),
        "olive_green_dark": (0.3333, 0.4196, 0.1843),
        "permanent_green": (0.0400, 0.7900, 0.1700),
        "sap_green": (0.1900, 0.5000, 0.0800),
        "sea_green": (0.1804, 0.5451, 0.3412),
        "sea_green_dark": (0.5608, 0.7373, 0.5608),
        "sea_green_medium": (0.2353, 0.7020, 0.4431),
        "sea_green_light": (0.1255, 0.6980, 0.6667),
        "spring_green": (0.0000, 1.0000, 0.4980),
        "spring_green_medium": (0.0000, 0.9804, 0.6039),
        "terre_verte": (0.2200, 0.3700, 0.0600),
        "viridian_light": (0.4300, 1.0000, 0.4400),
        "yellow_green": (0.6039, 0.8039, 0.1961),
        "aquamarine": (0.4980, 1.0000, 0.8314),
        "aquamarine_medium": (0.4000, 0.8039, 0.6667),
        "cyan": (0.0000, 1.0000, 1.0000),
        "cyan_white": (0.8784, 1.0000, 1.0000),
        "turquoise": (0.2510, 0.8784, 0.8157),
        "turquoise_dark": (0.0000, 0.8078, 0.8196),
        "turquoise_medium": (0.2824, 0.8196, 0.8000),
        "turquoise_pale": (0.6863, 0.9333, 0.9333),
        "alice_blue": (0.9412, 0.9725, 1.0000),
        "blue": (0.0000, 0.0000, 1.0000),
        "blue_light": (0.6784, 0.8471, 0.9020),
        "blue_medium": (0.0000, 0.0000, 0.8039),
        "cadet": (0.3725, 0.6196, 0.6275),
        "cobalt": (0.2400, 0.3500, 0.6700),
        "cornflower": (0.3922, 0.5843, 0.9294),
        "cerulean": (0.0200, 0.7200, 0.8000),
        "dodger_blue": (0.1176, 0.5647, 1.0000),
        "indigo": (0.0300, 0.1800, 0.3300),
        "manganese_blue": (0.0100, 0.6600, 0.6200),
        "midnight_blue": (0.0980, 0.0980, 0.4392),
        "navy": (0.0000, 0.0000, 0.5020),
        "peacock": (0.2000, 0.6300, 0.7900),
        "powder_blue": (0.6902, 0.8784, 0.9020),
        "royal_blue": (0.2549, 0.4118, 0.8824),
        "slate_blue": (0.4157, 0.3529, 0.8039),
        "slate_blue_dark": (0.2824, 0.2392, 0.5451),
        "slate_blue_light": (0.5176, 0.4392, 1.0000),
        "slate_blue_medium": (0.4824, 0.4078, 0.9333),
        "sky_blue": (0.5294, 0.8078, 0.9216),
        "sky_blue_deep": (0.0000, 0.7490, 1.0000),
        "sky_blue_light": (0.5294, 0.8078, 0.9804),
        "steel_blue": (0.2745, 0.5098, 0.7059),
        "steel_blue_light": (0.6902, 0.7686, 0.8706),
        "turquoise_blue": (0.0000, 0.7800, 0.5500),
        "ultramarine": (0.0700, 0.0400, 0.5600),
        "blue_violet": (0.5412, 0.1686, 0.8863),
        "cobalt_violet_deep": (0.5700, 0.1300, 0.6200),
        "magenta": (1.0000, 0.0000, 1.0000),
        "orchid": (0.8549, 0.4392, 0.8392),
        "orchid_dark": (0.6000, 0.1961, 0.8000),
        "orchid_medium": (0.7294, 0.3333, 0.8275),
        "permanent_red_violet": (0.8600, 0.1500, 0.2700),
        "plum": (0.8667, 0.6275, 0.8667),
        "purple": (0.6275, 0.1255, 0.9412),
        "purple_medium": (0.5765, 0.4392, 0.8588),
        "ultramarine_violet": (0.3600, 0.1400, 0.4300),
        "violet": (0.5600, 0.3700, 0.6000),
        "violet_dark": (0.5804, 0.0000, 0.8275),
        "violet_red": (0.8157, 0.1255, 0.5647),
        "violet_red_medium": (0.7804, 0.0824, 0.5216),
        "violet_red_pale": (0.8588, 0.4392, 0.5765),
    }

    _ReservedNames = CGK.cgnsnames
    _ReservedTypes = CGK.cgnstypes
    _SortedTypeList = CGK.sortedtypelist

    _UsualQueries = [
        # ---------------------------------------------------------------------------
        # INDENTATION IS SIGNIFICANT
        # ---------------------------------------------------------------------------
        # --- Search -----------------------------------------------------------
        # last two booleans: Update tree, has args
        (
            "001. Node name",
            "Search by",
            "RESULT=(NAME==ARGS[0])",
            """
Search by
Node name

Search all nodes with the exact NAME as argument.

The argument name need not to be a tuple or to have quotes,
all the following values are ok and would match the NAME <i>ZoneType</i>:

ZoneType
'ZoneType'
('ZoneType',)
""",
            False,
            True,
        ),
        (
            "002. Wildcard node name",
            "Search by",
            """import fnmatch
RESULT=fnmatch.fnmatchcase(NAME,ARGS[0])
""",
            """
Search by
Wildcard node name

Search all nodes with the wildcard NAME as argument.

Warning: the <b>argument name</b> should be quoted:

'BC*' is ok

BC* would fail
""",
            False,
            True,
        ),
        (
            "003. Node type",
            "Search by",
            "RESULT=(SIDSTYPE==ARGS[0])",
            """search all nodes with argument SIDS type.""",
            False,
            True,
        ),
        (
            "005. Node value",
            "Search by",
            """
from numpy import *
target=eval(ARGS[0])
if   (VALUE is None and target is None): RESULT=True
elif (VALUE is None)            : RESULT=False
elif (target.dtype!=VALUE.dtype): RESULT=False
elif (target.size!=VALUE.size):   RESULT=False
elif (target.shape!=VALUE.shape): RESULT=False
elif (target.tolist()==VALUE.tolist()): RESULT=True
else:                             RESULT=False
""",
            """search all nodes with argument value. The compare is performed
using a straightforward '==' and then relies on the Python/Numpy comparison
operator.""",
            False,
            True,
        ),
        (
            "010. Node with truncated data",
            "Search by",
            "if (PATH in SKIPS): RESULT=PATH",
            """search all nodes with truncated or unread data, for example if you have set
the maximum data argument for the load, or if you release the memory of a
node.""",
            False,
            False,
        ),
        (
            "004. Wildcard node type",
            "Search by",
            """
import fnmatch
RESULT=fnmatch.fnmatchcase(SIDSTYPE,ARGS[0])
""",
            """
Search by
Wildcard node type

Search all nodes with wildcard argument SIDS type.
Warning: the <b>argument type</b> should be quoted:

'Turb*' is ok

Turb* would fail
""",
            False,
            True,
        ),
        (
            "011. Non-MT UserDefinedData",
            "Search by",
            """
if (    (SIDSTYPE==CGK.UserDefinedData_ts)
    and (CGU.getValueDataType(NODE)!=CGK.MT)):
  RESULT=True
""",
            """
Search by
Valued UserDefinedData

Search all <b>UserDefinedData_t</b> nodes with a non-<b>MT</b> data type.

No argument.
""",
            False,
            False,
        ),
        # -----------------------------------------------------------------------------
        (
            "012. FamilyName",
            "Search by",
            """
if (SIDSTYPE in [CGK.FamilyName_ts, CGK.AdditionalFamilyName_ts]):
    RESULT=True
""",
            """
Search by
All <b>FamilyName_t</b> and <b>AdditionalFamilyname_t</b> nodes.
""",
            False,
            False,
        ),
        # -----------------------------------------------------------------------------
        (
            "013. FamilyName reference",
            "Search by",
            """
if ((SIDSTYPE in [CGK.FamilyName_ts, CGK.AdditionalFamilyName_ts]) and
    (VALUE.tostring().decode('ascii')==ARGS[0])):
    RESULT=True
""",
            """
Search by<br>
Reference to a FamilyName<br>
 
Search all <b>FamilyName</b> nodes with the arg string (plain).<br>
The string arg should be a valid Python string such as:<br>
 
'BLADE(1)'<br>
'Surface ext 1A'<br>
'Left/Wing/Flap'<br>
 
""",
            False,
            True,
        ),
        # -----------------------------------------------------------------------------
        (
            "014. Zones",
            "Search by",
            """
if (SIDSTYPE in [CGK.Zone_ts]):
    RESULT=True
""",
            """
Search by
All <b>Zone_t</b> nodes.
""",
            False,
            False,
        ),
        # -----------------------------------------------------------------------------
        (
            "015. Zones Structured",
            "Search by",
            """
if (SIDSTYPE in [CGK.Zone_ts]):
    t=CGU.hasChildName(NODE,CGK.ZoneType_s)
    if (t is None or CGU.stringValueMatches(t,CGK.Structured_s)):
      RESULT=True
""",
            """
Search by
All <b>Zone_t</b> with Structured <b>ZoneType</b> nodes.
""",
            False,
            False,
        ),
        # -----------------------------------------------------------------------------
        (
            "016. Zones Unstructured",
            "Search by",
            """
if (SIDSTYPE in [CGK.Zone_ts]):
    t=CGU.hasChildName(NODE,CGK.ZoneType_s)
    if (t is not None and CGU.stringValueMatches(t,CGK.Unstructured_s)):
      RESULT=True
""",
            """
Search by
All <b>Zone_t</b> with Unstructured <b>ZoneType</b> nodes.
""",
            False,
            False,
        ),
        # -----------------------------------------------------------------------------
        (
            "017. BCs",
            "Search by",
            """
if (SIDSTYPE in [CGK.BC_ts]):
    RESULT=True
""",
            """
Search by
All <b>BC_t</b> nodes.
""",
            False,
            False,
        ),
        # --- Replace
        # -----------------------------------------------------------------------------
        (
            "050. Valued UserDefinedData",
            "Replace",
            """
if (     (SIDSTYPE==CGK.UserDefinedData_ts)
     and (CGU.getValueDataType(NODE)!=CGK.MT)):
  NODE[3]=CGK.DataArray_ts
  RESULT=True
""",
            """
Replace
Valued UserDefinedData
 
Search all <b>UserDefinedData_t</b> nodes with a non-<b>MT</b> data type
and replace them as <b>DataArray_t</b>.""",
            False,
            False,
        ),
        (
            "051. Substitute Zone name",
            "Replace",
            """
l1=len(ARGS[0])
if ((SIDSTYPE==CGK.Zone_ts) and (NAME[:l1]==ARGS[0])):
  NODE[0]=ARGS[1]+NODE[0][l1:]
  RESULT=True

if (CGU.getValueDataType(NODE)==CGK.C1):
  v=VALUE.tostring().decode('ascii')
  if (v[:l1]==ARGS[0]):
    v=ARGS[1]+v[l1:]
    NODE[1]=CGU.setStringAsArray(v)
    RESULT=True
""",
            """
<h1>Replace</h1>
<h2>Substitute Zone name</h2>
<p>
Search all <b>Zone_t</b> nodes with a name pattern, then rename the
zone with the substitution pattern. Any other reference in the tree,
as a connectivity value for example, is subsitued as well.
<p>
Argument is a tuple with the first pattern to find and
the second as the subsitution pattern. For example:
<pre>
('domain.','zone#')
</pre>
""",
            True,
            True,
        ),
        # -----------------------------------------------------------------------------
        (
            "052. FamilySpecified BC type rewrite",
            "Replace",
            """
if ((SIDSTYPE in [CGK.FamilyName_ts, CGK.AdditionalFamilyName_ts]) and
    (VALUE.tostring().decode('ascii')==ARGS[0]) and (PARENT[3]==CGK.BC_ts)):
    PARENT[1]=CGU.setStringAsArray(CGK.FamilySpecified_s)
    RESULT=True
""",
            """
<h1>Replace</h1>
<h2>FamilySpecified BC type rewrite</h2>
<p>
Search all <b>FamilyName BC</b> nodes with the arg string (plain).<br>
The string arg should be a valid Python string such as:<br>

'BLADE(1)'<br>
'Surface ext 1A'<br>
'Left/Wing/Flap'<br>

<p>
Once found, the parent <b>BC_t</b> value is forced to <b>FamilySpecified</b>

""",
            True,
            True,
        ),
        # --- Find Elements_t
        (
            "020. Elements",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=True
""",
            """Find all <b>Elements_t</b> nodes """,
            False,
            False,
        ),
        (
            "021. Elements QUAD",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=VALUE[0] in (CGK.QUAD_4, CGK.QUAD_8, CGK.QUAD_9)
""",
            """Find all <b>Elements_t</b> nodes of type <b>QUAD</b>""",
            False,
            False,
        ),
        (
            "022. Elements TRI",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=VALUE[0] in (CGK.TRI_3, CGK.TRI_6)
""",
            """Find all <b>Elements_t</b> nodes of type <b>TRI</b>""",
            False,
            False,
        ),
        (
            "023. Elements NGON",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=VALUE[0] in (CGK.NGON_n,)
""",
            """Find all <b>Elements_t</b> nodes of type <b>NGON_n</b>""",
            False,
            False,
        ),
        (
            "024. Elements HEXA",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=VALUE[0] in (CGK.HEXA_8, CGK.HEXA_20, CGK.HEXA_27)
""",
            """Find all <b>Elements_t</b> nodes of type <b>HEXA</b>""",
            False,
            False,
        ),
        (
            "025. Elements TETRA",
            "Find Elements_t",
            """if (SIDSTYPE==CGK.Elements_ts):
  RESULT=VALUE[0] in (CGK.TETRA_4, CGK.TETRA_10)
""",
            """Find all <b>Elements_t</b> nodes of type <b>TETRA</b>""",
            False,
            False,
        ),
        # --- External Tools
        (
            "030. Create Cartesian Zone",
            "External Tools",
            """
if (SIDSTYPE==CGK.CGNSTree_ts):
    import Generator.PyTree as G
    z=G.cart((0.,0.,0.), (0.1,0.1,0.2), (10,11,12))
    b=None
    base='BASE'
    if (len(ARGS)>0):
      base=ARGS[0]
      b=CGU.hasChildName(NODE,base)
    if (b is None):
      base=CGU.checkUniqueChildName(NODE,base)
      b=CGL.newCGNSBase(NODE,base,3,3)
    CGU.addChild(b,z)
""",
            """Example of Cartesian zone creation using Cassiopee.
The first argument is the base name, if ommitted a name is generated.""",
            True,
            True,
        ),
        (
            "031. Bounding boxes",
            "External Tools",
            """
if (SIDSTYPE==CGK.Zone_ts):
    import Generator as G
    RESULT=G.bbox(NODE)
""",
            """Example of Bounding box computation using Cassiopee""",
        ),
        (
            "100. .Solver#Compute children",
            "Edit filters",
            """
if (PARENT[0]=='.Solver#Compute'):
    RESULT=PATH
""",
            """Selects all children nodes of the .Solver#Compute elsA userdefined node""",
            False,
            False,
        ),
        (
            "101. ReferenceState children",
            "Edit filters",
            """
if (PARENT[0]=='ReferenceState'):
    RESULT=PATH
""",
            """Selects all children nodes of the ReferenceState node""",
            False,
            False,
        ),
        (
            "102. .Solver#Param children",
            "Edit filters",
            """
if (PARENT[0]=='.Solver#Param'):
   RESULT=PATH
""",
            """Selects all children nodes of the .Solver#Param elsA userdefined node""",
            False,
            False,
        ),
    ]

    # -----------------------------------------------------------------
    @classmethod
    def _setOption(cls, name, value):
        setattr(cls, name, value)

    @classmethod
    def _writeFile(cls, tag, name, udata, filename, prefix=""):
        gdate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        s = """# %s - %s - Generated %s\n# coding: utf-8\n%s\nimport PyQt5\n%s=""" % (
            cls._ToolName,
            tag,
            gdate,
            prefix,
            name,
        )
        if isinstance(udata, dict):
            s += """{\n"""
            for k in udata:
                #          print 'K',k,'V',udata[k],'T',type(udata[k])
                if k[0] != "_":
                    val = "%s" % str(udata[k])
                    if isinstance(udata[k], str):
                        val = 'u"""%s"""' % repr(udata[k]).lstrip("u'").lstrip(
                            "'"
                        ).rstrip(
                            "'"
                        )  # Error
                    elif isinstance(udata[k], bytes):
                        val = 'u"""%s"""' % repr(udata[k].decode("utf-8")).lstrip(
                            "u'"
                        ).lstrip("'").rstrip("'")
                    if not isinstance(k, str):
                        uk = "u'%s'" % repr(k.decode("utf-8")).lstrip("u'").lstrip(
                            "'"
                        ).rstrip("'")
                    else:
                        uk = "u'%s'" % repr(k).lstrip("u'").lstrip("'").rstrip("'")
                    s += """%s:%s,\n""" % (uk, val)
            s += """}\n\n# --- last line\n"""
        elif isinstance(udata, list):
            s += """[\n"""
            for k in udata:
                s += """%s,\n""" % (k)
            s += """]\n\n# --- last line\n"""
        cls._crpath(filename)
        with open(filename, "w+") as f:
            f.write(s)

    @classmethod
    def _readFile(cls, name, filename):
        dpath = tempfile.mkdtemp()
        if not op.exists(filename):
            return None
        try:
            copyOneFile(filename, "%s/%s.py" % (dpath, name))
        except IOError:
            removeSubDirAndFiles(dpath)
            return None
        sprev = sys.path
        sys.path = [dpath] + sys.path
        try:
            fp, pathname, description = imp.find_module(name)
        except IndexError:  # ImportError:
            return None
        try:
            mod = imp.load_module(name, fp, pathname, description)
        finally:
            if fp:
                fp.close()
        removeSubDirAndFiles(dpath)
        sys.path = sprev
        return mod

    @classmethod
    def _crpath(cls, path):
        p = op.dirname(path)
        if op.exists(p):
            return True
        os.makedirs(p)

    @classmethod
    def _trpath(cls, path):
        return op.normpath(op.expanduser(op.expandvars(path)))

    @classmethod
    def _writeHistory(cls, control):
        filename = cls._trpath(cls._HistoryFileName)
        cls._writeFile("History", "history", control._history, filename)

    @classmethod
    def _readHistory(cls, control):
        filename = cls._trpath(cls._HistoryFileName)
        m = cls._readFile("history", filename)
        if m is None:
            return None
        try:
            return m.history
        except:
            return None

    @classmethod
    def _writeOptions(cls, control):
        filename = cls._trpath(cls._OptionsFileName)
        cls._writeFile("User options", "options", control._options, filename)

    @classmethod
    def _readOptions(cls, control):
        filename = cls._trpath(cls._OptionsFileName)
        m = cls._readFile("options", filename)
        if m is None:
            return None
        try:
            return m.options
        except:
            return None

    @classmethod
    def _writeQueries(cls, control, q):
        filename = cls._trpath(op.join(cls.QueriesDirectory, cls._QueriesDefaultFile))
        cls._writeFile("User queries", "queries", q, filename, cls.Q_FILE_PRE)

    @classmethod
    def _readQueries(cls, control):
        filename = cls._trpath(op.join(cls.QueriesDirectory, cls._QueriesDefaultFile))
        m = cls._readFile("queries", filename)
        if m is None:
            return None
        try:
            return m.queries
        except:
            return None

    @classmethod
    def _writeFunctions(cls, control, f):
        filename = cls._trpath(
            op.join(cls.FunctionsDirectory, cls._FunctionsDefaultFile)
        )
        cls._writeFile("User functions", "functions", f, filename, cls.Q_FILE_PRE)

    @classmethod
    def _readFunctions(cls, control):
        filename = cls._trpath(
            op.join(cls.FunctionsDirectory, cls._FunctionsDefaultFile)
        )
        m = cls._readFile("functions", filename)
        if m is None:
            return None
        try:
            return m.Q7UserFunction
        except:
            return None

    def __init__(self):
        pass

    def __getitem__(self, name):
        if name[0] != "_":
            return Q7OptionContext.__dict__[name]
        return None

    def __setitem__(self, name, value):
        if name[0] != "_":
            setattr(Q7OptionContext, name, value)
        return None

    def __iter__(self):
        for o in dir(self):
            if o[0] != "_":
                yield o

    def _nextName(self):
        for o in dir(self):
            if o[0] != "_":
                yield o


# -----------------------------------------------------------------
