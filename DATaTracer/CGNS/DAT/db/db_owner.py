#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# ------------------------------------------------------------
# pyDAX - DBMS schema - values
# ------------------------------------------------------------
#
from CGNS.DAT.exceptions      import *
#
defaultOwners=[
  """INSERT \
     INTO cgnsOwner (organisation,name,site,description) \
     VALUES (?,?,?,?)""",
  (
    ("UNKNOWN",    "UNKNOWN",    "X",   "<Value has not been set>"),
    ("ONERA",      "DSNA/CS2A",  "E",   "Code Validation"),    
    ("ONERA",      "DSNA/ACOU",  "E",   "Acoustics"),
    ("ONERA",      "DAAP/H2T" ,  "E",   "Applied Dept."),
    ("DLR",        "AS",         "BS",  "German Dept."),
    ("Eurocopter", "-",          "F",   "Customer"),
    ("Eurocopter", "-",          "D",   "Customer"),
  )
]    
#
valuesList=[
 ['cgnsOwnerList', defaultOwners],
]
# --- FOR TEST WE USE SITE INSTEAD OF USER NAME ***
#
def getOwner(db,name):
  stmt="""select id from cgnsOwner where site="%s" """%name
  try:
    oid=db.runSQL(stmt)[0][0]
  except IndexError:
      raise DAXNoSuchOwner(name)
  return oid
#
