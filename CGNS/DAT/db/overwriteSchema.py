#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
# pyDAX - DBMS schema
# ------------------------------------------------------------
#
import CGNS.DAT.db.db01
import CGNS.DAT.db.db02
import CGNS.DAT.db.db03
import CGNS.DAT.db.db_owner
import CGNS.DAT.db.db_test
import CGNS.DAT.db.connect
import CGNS.DAT.db.dbdrivers
from   CGNS.DAT.exceptions import *
from   CGNS.DAT.utils import *


#
# --- create (drop) tables
#
def dropTableList(tablelist, db):
    for tb in tablelist:
        pdebug("Drop %s" % tb[0])
        stmt = "DROP TABLE IF EXISTS %s" % tb[0]
        db.runSQL(stmt)


#
def dropIndexList(indexlist, db):
    for tb in indexlist:
        pdebug("Drop %s" % tb[0])
        stmt = "DROP INDEX IF EXISTS %s" % tb[0]
        db.runSQL(stmt)


#
def createTableList(tablelist, db):
    for tb in tablelist:
        pdebug("Create %s" % tb[0])
        db.runSQL(tb[1])


#
# --- Add values in tables
#
def addValues(valueslist, db):
    for vlst in valueslist:
        pdebug("Add %s" % vlst[0])
        vl = vlst[1]
        stmt = vl[0]  # statement is first element
        db.runSQL(stmt, values=vl[1])  # then list of values


#
def getOwner(name, db):
    stmt = """select id from cgnsOwner where name="%s" """
    r = db.runSQL(stmt % name)
    if not r: raise DAXNoSuchOwner(db.name, name)
    return r[0]


#
def overWrite(dbname):
    name, u, p = dbname.split(':')
    db = CGNS.DAT.db.dbdrivers.daxDriverDefault(name, user=u, passwd=p)
    #
    dropTableList(CGNS.DAT.db.db01.tableList, db)
    dropTableList(CGNS.DAT.db.db02.tableList, db)
    dropTableList(CGNS.DAT.db.db03.tableList, db)
    dropIndexList(CGNS.DAT.db.db01.indexList, db)
    dropIndexList(CGNS.DAT.db.db02.indexList, db)
    dropIndexList(CGNS.DAT.db.db03.indexList, db)
    #
    createTableList(CGNS.DAT.db.db01.tableList, db)
    createTableList(CGNS.DAT.db.db02.tableList, db)
    createTableList(CGNS.DAT.db.db03.tableList, db)
    #
    addValues(CGNS.DAT.db.db_owner.valuesList, db)
    addValues(CGNS.DAT.db.db_test.valuesList, db)
    #
    del db
    #


#
if (__name__ == "__main__"):
    import sys

    overWrite(sys.argv[-1])

#
# -- last line
