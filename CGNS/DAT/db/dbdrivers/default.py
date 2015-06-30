#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
# pyCGNS.DAT - DBMS schema - Manages connection (high level)
# ------------------------------------------------------------
#
# DBMS connection and drivers
# - The DB2 API actually does the job...
#
import CGNS.DAT.utils           as dxUT
import CGNS.DAT.exceptions      as dxEX
#
# ------------------------------------------------------------
class daxDriver:
  def __init__(self,base,user,passwd):
    self.name=base
    self._user=user
    self._passwd=passwd
  def __del__(self):
    dxUT.ptrace("SQL # (close)")
    if self._db:
      self._db.close()
  def cursor(self):
    return self._db.cursor()
  def runSQLgeneric(self,stmt):
    cs=self.cursor()
    dxUT.ptrace("SQL # %s"%stmt)
    cs.execute(stmt)
    return cs.fetchall()    
  def runSQLgenericMany(self,stmt,values):
    cs=self.cursor()
    dxUT.ptrace("SQL # %s"%stmt)
    cs.executemany(stmt,values)
    return cs.fetchall()

# -----------------------------------------------------------------------------
# No DB found
# -----------------------------------------------------------------------------
class daxDriverNoneDumbDB():
  def __init__(self):
    pass
  def close(self):
    pass
  def cursor(self):
    return None
  
class daxDriverNone(daxDriver):
  def __init__(self,base,user='',passwd=''):
    self._db = daxDriverNoneDumbDB()
    daxDriver.__init__(self,base,user,passwd)
  def commit(self):
    pass
  def rollback(self):
    pass
  def runSQL(self,stmt,commit=0,values=None):
    pass

daxDriverDefault=daxDriverNone

# -----------------------------------------------------------------------------
# mySQL - *** NOT USABLE ANYMORE BECAUSE SCHEMA REALLY IS DIFFERENT
#         *** MOVE TO SQLITE (below)
# -----------------------------------------------------------------------------
__d_mysql=0
# try:
#   import MySQLdb
#   import MySQLdb.cursors
#   __d_mysql=1
# except:
#   pass

# if __d_mysql:
#   class daxDriverMySQL(daxDriver):
#     def __init__(self,base,user='',passwd=''):
#       self._db = None
#       daxDriver.__init__(self,base,user,passwd)
#       try:
#         self._db = MySQLdb.Connect(db=base,user=user,passwd=passwd,
#                                    cursorclass=MySQLdb.cursors.CursorNW)
#       except MySQLdb.OperationalError, tp:
#         #dxUT.perror(tp,1)
#         raise dxEX.DAXSystemError(str(tp))
#       except MySQLdb.NotSupportedError, tp:
#         pass
#         #raise dxEX.DAXSystemError(str(tp))
#     def commit(self):
#       self._db.commit()
#     def rollback(self):
#       try:
#         self._db.rollback()
#       except MySQLdb.ProgrammingError, tp:
#         pass
#       except MySQLdb.NotSupportedError, tp:
#         pass
#     def runSQL(self,stmt,commit=0,values=None):
#       r=None
#       try:
#         if (values): r=self.runSQLgenericMany(stmt,values)
#         else:        r=self.runSQLgeneric(stmt)
#         if (commit):
#           self._db.commit()
#         return r
#       except MySQLdb.NotSupportedError, tp:
#         pass
#         #raise dxEX.DAXSystemError(str(tp))
#       except MySQLdb.ProgrammingError, tp:
#         #dxUT.perror(tp)
#         if values:
#           raise dxEX.DAXSystemError(str((tp,stmt,values)))
#           #dxUT.perror(stmt)
#           #dxUT.perror(values,1)
#         raise dxEX.DAXSystemError(str((tp,stmt)))
#         #dxUT.perror(stmt,1)      
#       except MySQLdb.OperationalError, tp:
#         #dxUT.perror(tp,1)
#         raise dxEX.DAXSystemError(str(tp))
#   daxDriverDefault=daxDriverMySQL

# -----------------------------------------------------------------------------
# sqlite3
# -----------------------------------------------------------------------------
__d_sqlite3=0
try:
  import sqlite3
  __d_sqlite3=1
except:
  pass

if __d_sqlite3:
  class daxDriverSqlite3(daxDriver):
    def __init__(self,base,user='',passwd=''):
      self._db = None
      daxDriver.__init__(self,base,user,passwd)
      self._db = sqlite3.connect(database=base)
    def commit(self):
      self._db.commit()
    def rollback(self):
      self._db.rollback()
    def runSQL(self,stmt,commit=0,values=None):
      r=None
      dxUT.perror(stmt)
      if (values): r=self.runSQLgenericMany(stmt,values)
      else:        r=self.runSQLgeneric(stmt)
      if (commit):
        self._db.commit()
      return r
  daxDriverDefault=daxDriverSqlite3

# --- last line
