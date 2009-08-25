# ------------------------------------------------------------
# pyDAX - DBMS schema - values
# ------------------------------------------------------------
# $Id: db_owner.py 30 2003-08-04 14:40:17Z mpoinot $
#
from   DAX.exceptions      import *
#
defaultOwners=[
  """INSERT \
     INTO cgnsOwner (organisation,name,site,description) \
     VALUES (%s,%s,%s,%s)""",
  (
    ("UNKNOWN",    "UNKNOWN",    "X",   "<Value has not been set>"),
    ("ONERA",      "DSNA/ELSA",  "E",   "Code Validation"),    
    ("ONERA",      "DSNA/MNEI",  "E",   "Numerical Methods"),
    ("ONERA",      "DAAP/H2T" ,  "E",   "CHANCE project"),
    ("DLR",        "AS",         "BS",  "CHANCE project"),
    ("Eurocopter", "-",          "F",   "CHANCE project"),
    ("Eurocopter", "-",          "D",   "CHANCE project"),
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
