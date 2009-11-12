# ------------------------------------------------------------
# pyDAX - DBMS schema - Manages connection (high level)
# ------------------------------------------------------------
# $Id: connect.py 25 2003-07-18 13:15:37Z mpoinot $
#
# DBMS connection and drivers
# - The DB2 API actually does the job...
#
import DAX.utils           as dxUT
import DAX.exceptions      as dxEX
import DAX.preparedQueries as dxPQ
import DAX.db.parseCGNS    as dxPS
import DAX.db.foreignKeys  as dxFK
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
#
import MySQLdb
import MySQLdb.cursors 
# ------------------------------------------------------------
class daxDriverMySQL(daxDriver):
  def __init__(self,base,user='',passwd=''):
    self._db = None
    daxDriver.__init__(self,base,user,passwd)
    try:
      self._db = MySQLdb.Connect(db=base,user=user,passwd=passwd,
                                 cursorclass=MySQLdb.cursors.CursorNW)
    except MySQLdb.OperationalError, tp:
      #dxUT.perror(tp,1)
      raise dxEX.DAXSystemError(str(tp))
    except MySQLdb.NotSupportedError, tp:
      pass
      #raise dxEX.DAXSystemError(str(tp))
  def commit(self):
    self._db.commit()
  def rollback(self):
    try:
      self._db.rollback()
    except MySQLdb.ProgrammingError, tp:
      pass
    except MySQLdb.NotSupportedError, tp:
      pass
  def runSQL(self,stmt,commit=0,values=None):
    r=None
    try:
      if (values): r=self.runSQLgenericMany(stmt,values)
      else:        r=self.runSQLgeneric(stmt)
      if (commit):
        self._db.commit()
      return r
    except MySQLdb.NotSupportedError, tp:
      pass
      #raise dxEX.DAXSystemError(str(tp))
    except MySQLdb.ProgrammingError, tp:
      #dxUT.perror(tp)
      if values:
        raise dxEX.DAXSystemError(str((tp,stmt,values)))
        #dxUT.perror(stmt)
        #dxUT.perror(values,1)
      raise dxEX.DAXSystemError(str((tp,stmt)))
      #dxUT.perror(stmt,1)      
    except MySQLdb.OperationalError, tp:
      #dxUT.perror(tp,1)
      raise dxEX.DAXSystemError(str(tp))
#
daxDriverDefault=daxDriverMySQL
#
