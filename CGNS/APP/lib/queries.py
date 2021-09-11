#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#

def asQuery(f):
    def prepostquery(node, parent, tree, links, skips, path, args, selected):
        import CGNS.PAT.cgnskeywords as CGK
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnslib as CGL
        import CGNS.MAP as CGM
        import numpy

        QueryNoException = True

        class Context(object):
            pass

        C = Context()
        C.NODE = node
        C.PARENT = parent
        C.NAME = node[0]
        C.VALUE = node[1]
        C.CGNSTYPE = node[3]
        C.CHILDREN = node[2]
        C.TREE = tree
        C.PATH = path
        C.LINKS = links
        C.SKIPS = skips
        C.USER = args
        C.ARGS = args
        C.SELECTED = selected
        C.RESULT_LIST = [False]

        if (QueryNoException):
            RESULT = f(C)
        else:
            try:
                RESULT = f(C)
                if (RESULT not in [True, False]): RESULT = False
                RESULT_LIST[0] = RESULT
            except Exception:
                RESULT_LIST[0] = False

        return RESULT

    return prepostquery


# -----------------------------------------------------------------
def parseAndSelect(tree, node, parent, links, skips, path, query, args, selected,
                   result):
    path = path + '/' + node[0]
    print('ARGS ', args)
    Q = query(node, parent, tree, links, skips, path, args, selected)
    R = []
    if (Q):
        if (result):
            R = [Q]
        else:
            R = [path]
    for C in node[2]:
        R += parseAndSelect(tree, C, node, links, skips, path, query, args, selected,
                            result)
    return R


# -----------------------------------------------------------------
def runQuery(tree, links, paths, query, args, selected=[], mode=True):
    """Recursively applies a function on all nodes of a tree, breadth-first
    parse, results returned in a list (breadth-first order of the list).

    tree,links,paths: the three restults returned by the CGNS.MAP.load call
    query: the query funciton as a name (string) or a callable object (such as
    a function)
    args: a tuple of args to pass to the query (for example a node name to
    look for)
    selected: the RETURNED list of values
    mode: True (defaut) a boolean list is returned for every node in the tree,
    False a list of the (True) paths is returned
    
    The match between results and paths can be performed with help of the
    breadth-first paths order function in CGNS.PAT
    """
    v = None
    try:
        try:
            if (args): v = eval(args)
        except TypeError:
            v = args
        if ((v is not None) and (type(v) not in [tuple, list])): v = (v,)
    except NameError:
        v = (str(args),)
    except:
        pass
    _args = v
    if (type(query) in [str,]): query = eval(query)
    result = parseAndSelect(tree, tree, [None, None, [], None], links, paths, '',
                            query, _args, selected, mode)
    return result
