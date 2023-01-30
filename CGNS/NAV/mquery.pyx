#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from ..NAV.moption import Q7OptionContext as OCTXT

import numpy

SCRIPT_PATTERN = """#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  CGNS.NAV - GENERATED FILE - %(Q_VAR_DATE)s
#  query: %(Q_VAR_QUERYNAME)s
# -----------------------------------------------------------------
from .. import MAP as CGM

FILE='%(Q_VAR_TREE_FILE)s'
SCRIPT=\"\"\"%(Q_VAR_QUERY_SCRIPT)s\"\"\"
ARGS=\"\"\"%(Q_VAR_QUERY_ARGS)s\"\"\"
SELECTED=\"\"\"%(Q_VAR_SELECTED)s\"\"\"

# -----------------------------------------------------------------
SCRIPT_PRE=\"\"\"%(Q_VAR_SCRIPT_PRE)s\"\"\"
SCRIPT_POST=\"\"\"%(Q_VAR_SCRIPT_POST)s\"\"\"
# -----------------------------------------------------------------
def evalScript(node,parent,tree,links,skips,path,val,args,selected):
    l=locals()
    l['%(Q_VAR_RESULT_LIST)s']=[False]
    l['%(Q_VAR_PARENT)s']=parent
    l['%(Q_VAR_NAME)s']=node[0]
    l['%(Q_VAR_VALUE)s']=node[1]
    l['%(Q_VAR_CGNSTYPE)s']=node[3]
    l['%(Q_VAR_CHILDREN)s']=node[2]
    l['%(Q_VAR_TREE)s']=tree
    l['%(Q_VAR_LINKS)s']=links
    l['%(Q_VAR_SKIPS)s']=skips
    l['%(Q_VAR_SELECTED)s']=selected
    l['%(Q_VAR_PATH)s']=path
    if (args is None): args=()
    l['%(Q_VAR_USER)s']=args
    l['%(Q_VAR_NODE)s']=node
    pre=SCRIPT_PRE+val+SCRIPT_POST
    try:
      eval(compile(pre,'<string>','exec'),globals(),l)
    except Exception:
      l['%(Q_VAR_RESULT_LIST)s'][0]=False
    RESULT=l['%(Q_VAR_RESULT_LIST)s'][0]
    return RESULT
# -----------------------------------------------------------------
def parseAndSelect(tree,node,parent,links,skips,path,script,args,selected,
                   result):
    path=path+'/'+node[0]
    Q=evalScript(node,parent,tree,links,skips,path,script,args,selected)
    R=[]
    if (Q):
        if (result):
            R=[Q]
        else:
            R=[path]
    for C in node[2]:
        R+=parseAndSelect(tree,C,node,links,skips,path,script,args,selected,
                          result)
    return R

# -----------------------------------------------------------------
def run(tree,links,skips,mode,args,script,selected):
    v=None
    try:
        if (args): v=eval(args)
        if ((v is not None) and (type(v)!=tuple)): v=(v,)
    except NameError:
        v=(str(args),)
    except:
        pass
    _args=v
    result=parseAndSelect(tree,tree,[None,None,[],None],links,skips,'',
                          script,_args,selected,mode)
    return result
# -----------------------------------------------------------------
(t,l,p)=CGM.load(FILE)
print run(t,l,p,True,ARGS,SCRIPT,SELECTED)

# -----------------------------------------------------------------
"""


# -----------------------------------------------------------------
def sameVal(n, v):
    if n is None:
        if v is None:
            return True
        return False
    if n.dtype.char in ['S', 'c']:
        return n.tostring().decode('ascii') == v
    if n.dtype.char in ['d', 'f', 'i', 'l', 'q']:
        return n.flat[0] == v
    if n == v:
        return True
    return False


# -----------------------------------------------------------------
def sameValType(n, v):
    if isinstance(n, numpy.ndarray):
        return False
    if n.dtype.char == v:
        return True
    return False


# -----------------------------------------------------------------
def evalScript(node, parent, tree, links, skips, path, val, args, selected):
    l = locals()
    l[OCTXT.Q_VAR_RESULT_LIST] = [False]
    l[OCTXT.Q_VAR_PARENT] = parent
    l[OCTXT.Q_VAR_NAME] = node[0]
    l[OCTXT.Q_VAR_VALUE] = node[1]
    l[OCTXT.Q_VAR_CGNSTYPE] = node[3]
    l[OCTXT.Q_VAR_CHILDREN] = node[2]
    l[OCTXT.Q_VAR_TREE] = tree
    l[OCTXT.Q_VAR_LINKS] = links
    l[OCTXT.Q_VAR_SKIPS] = skips
    l[OCTXT.Q_VAR_PATH] = path
    l[OCTXT.Q_VAR_SELECTED] = selected
    if args is None:
        args = ()
    l[OCTXT.Q_VAR_USER] = args
    l[OCTXT.Q_VAR_NODE] = node
    pre = OCTXT.Q_SCRIPT_PRE + val + OCTXT.Q_SCRIPT_POST
    if OCTXT.QueryNoException:
        eval(compile(pre, '<string>', 'exec'), globals(), l)
    else:
        try:
            eval(compile(pre, '<string>', 'exec'), globals(), l)
        except Exception:
            l[OCTXT.Q_VAR_RESULT_LIST][0] = False
    RESULT = l[OCTXT.Q_VAR_RESULT_LIST][0]
    return RESULT


# -----------------------------------------------------------------
def parseAndSelect(tree, node, parent, links, skips, path, script, args, selected,
                   result):
    path = path + '/' + node[0]
    Q = evalScript(node, parent, tree, links, skips, path, script, args, selected)
    R = []
    if Q:
        if result:
            R = [Q]
        else:
            R = [path]
    for C in node[2]:
        R += parseAndSelect(tree, C, node, links, skips, path, script, args,
                            selected, result)
    return R


# -----------------------------------------------------------------
class Q7QueryEntry(object):
    def __init__(self, name, group=None, script='', doc='',
                 update=False, hasargs=False):
        self._name = name
        self._group = group
        self._script = script
        self._doc = doc
        self._update = update
        self._hasargs = hasargs

    @property
    def name(self):
        return self._name

    @property
    def group(self):
        return self._group

    @property
    def doc(self):
        return self._doc

    @property
    def hasArgs(self):
        return self._hasargs

    @property
    def script(self):
        return self._script

    def requireTreeUpdate(self):
        return self._update

    def setRequireTreeUpdate(self, value):
        self._update = value

    def setScript(self, value):
        self._script = value

    def setDoc(self, value):
        self._doc = value

    def __str__(self):
        s = '("%s","%s","%s","""%s""",%s)' % \
            (self.name, self.group, self._script, self._doc, self._update)
        return s

    def run(self, tree, links, skips, mode, args, selected=None):
        if selected is None:
            selected = []
        v = None
        try:
            if isinstance(args, numpy.ndarray):
                v = ('numpy.' + repr(args),)
            else:
                if args not in [None, [], ()]:
                    v = eval(args)
                if (v is not None) and not isinstance(v, tuple):
                    v = (v,)
        except NameError as e:
            v = (str(args),)
        except Exception as e:
            print(e)
        self._args = v
        result = parseAndSelect(tree, tree, [None, None, [], None], links, skips, '',
                                self._script, self._args, selected, mode)
        return result

    def getFullScript(self, filename, text, args):
        datadict = {}
        datadict['Q_VAR_DATE'] = '00/00/00'
        datadict['Q_VAR_QUERYNAME'] = '%s/%s' % (self._group, self._name)
        datadict['Q_VAR_SCRIPT_PRE'] = OCTXT.Q_SCRIPT_PRE
        datadict['Q_VAR_SCRIPT_POST'] = OCTXT.Q_SCRIPT_POST
        datadict['Q_VAR_RESULT_LIST'] = OCTXT.Q_VAR_RESULT_LIST
        datadict['Q_VAR_PARENT'] = OCTXT.Q_VAR_PARENT
        datadict['Q_VAR_NAME'] = OCTXT.Q_VAR_NAME
        datadict['Q_VAR_VALUE'] = OCTXT.Q_VAR_VALUE
        datadict['Q_VAR_CGNSTYPE'] = OCTXT.Q_VAR_CGNSTYPE
        datadict['Q_VAR_CHILDREN'] = OCTXT.Q_VAR_CHILDREN
        datadict['Q_VAR_TREE'] = OCTXT.Q_VAR_TREE
        datadict['Q_VAR_PATH'] = OCTXT.Q_VAR_PATH
        datadict['Q_VAR_LINKS'] = OCTXT.Q_VAR_LINKS
        datadict['Q_VAR_SKIPS'] = OCTXT.Q_VAR_SKIPS
        datadict['Q_VAR_USER'] = OCTXT.Q_VAR_USER
        datadict['Q_VAR_NODE'] = OCTXT.Q_VAR_NODE
        datadict['Q_VAR_SELECTED'] = OCTXT.Q_VAR_SELECTED
        datadict['Q_VAR_QUERY_SCRIPT'] = text
        datadict['Q_VAR_QUERY_ARGS'] = args
        datadict['Q_VAR_TREE_FILE'] = filename
        script = SCRIPT_PATTERN % datadict
        return script

# -----------------------------------------------------------------
