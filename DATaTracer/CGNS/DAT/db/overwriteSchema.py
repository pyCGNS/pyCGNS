# ------------------------------------------------------------
# pyDAX - DBMS schema
# ------------------------------------------------------------
# $Id: overwriteSchema.py 29 2003-07-30 13:26:19Z mpoinot $
#
import DAX.db.db01
import DAX.db.db02
import DAX.db.db03
import DAX.db.db_owner
import DAX.db.db_test
import DAX.db.connect
import DAX.db.dbdrivers
from   DAX.exceptions  import *
from   DAX.utils       import *
#
# --- create (drop) tables
#
def dropTableList(tablelist,db):
  for tb in tablelist:
    pdebug("Drop %s"%tb[0])
    stmt = "DROP TABLE IF EXISTS %s"%tb[0]
    db.runSQL(stmt)
#
def createTableList(tablelist,db):
  for tb in tablelist:
    pdebug("Create %s"%tb[0])
    db.runSQL(tb[1])
#
# --- Add values in tables
#
def addValues(valueslist,db):
  for vlst in valueslist:
    pdebug("Add %s"%vlst[0])
    vl=vlst[1]
    stmt=vl[0]                   # statement is first element
    db.runSQL(stmt,values=vl[1]) # then list of values
#
def getOwner(name,db):
  stmt="""select id from cgnsOwner where name="%s" """
  r=db.runSQL(stmt%name)
  if not r: raise DAXNoSuchOwner(db.name,name)
  return r[0]
#
def overWrite(dbname):
  name,u,p=dbname.split(':')
  db=DAX.db.dbdrivers.daxDriverDefault(name,user=u,passwd=p)
  #
  dropTableList(DAX.db.db01.tableList,db)
  dropTableList(DAX.db.db02.tableList,db)
  dropTableList(DAX.db.db03.tableList,db)  
  #
  createTableList(DAX.db.db01.tableList,db)
  createTableList(DAX.db.db02.tableList,db)
  createTableList(DAX.db.db03.tableList,db)
  #
  addValues(DAX.db.db_owner.valuesList,db)
  addValues(DAX.db.db_test.valuesList,db)  
  #
  del db
  #
#
if (__name__ == "__main__"):
  import sys
  overWrite(sys.argv[-1])
    
#
# -- last line
