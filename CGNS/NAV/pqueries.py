#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#


def asQuery(f):
    def prepostquery(node, parent, tree, links, skips, path, args, selected):
        global CGK
        global CGU
        global CGL
        global numpy
        global NODE
        global PARENT
        global NAME
        global VALUE
        global CGNSTYPE
        global CHILDREN
        global TREE
        global PATH
        global LINKS
        global SKIPS
        global RESULT
        global USER
        global SELECTED
        global RESULT_LIST

        import CGNS.PAT.cgnskeywords as CGK
        import CGNS.PAT.cgnsutils as CGU
        import CGNS.PAT.cgnslib as CGL
        import CGNS.MAP as CGM
        import numpy

        QueryNoException = True

        NODE = node
        PARENT = parent
        NAME = node[0]
        VALUE = node[1]
        CGNSTYPE = node[3]
        CHILDREN = node[2]
        TREE = tree
        PATH = path
        LINKS = links
        SKIPS = skips
        USER = args
        SELECTED = selected
        RESULT_LIST = [False]

        if QueryNoException:
            RESULT = f()
        else:
            try:
                RESULT = f()
                if RESULT not in [True, False]:
                    RESULT = False
                RESULT_LIST[0] = RESULT
            except Exception:
                RESULT_LIST[0] = False

        return RESULT

    return prepostquery


# -----------------------------------------------------------------
def parseAndSelect(
    tree, node, parent, links, skips, path, query, args, selected, result
):
    path = path + "/" + node[0]
    Q = query(node, parent, tree, links, skips, path, args, selected)
    R = []
    if Q:
        if result:
            R = [Q]
        else:
            R = [path]
    for C in node[2]:
        R += parseAndSelect(
            tree, C, node, links, skips, path, query, args, selected, result
        )
    return R


# -----------------------------------------------------------------
def runQuery(tree, links, paths, query, args, selected=None, mode=True):
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
    if selected is None:
        selected = []
    v = None
    try:
        if args:
            v = eval(args)
        if (v is not None) and not isinstance(v, tuple):
            v = (v,)
    except NameError:
        v = (str(args),)
    except:
        pass
    _args = v
    if isinstance(query, (str, bytes)):
        query = eval(query)
    result = parseAndSelect(
        tree,
        tree,
        [None, None, [], None],
        links,
        paths,
        "",
        query,
        _args,
        selected,
        mode,
    )
    return result
