#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release:  $
#  -------------------------------------------------------------------------
from CGNS import pyCGNSconfig

try:
  if (not pyCGNSconfig.HAS_MLL):
    import sys
    print "### pyCGNS: NOT built with ADF/Python binding"
    print "### pyCGNS: Cannot perform ADF tests"
    sys.exit(1)
except KeyError:
    print "### pyCGNS: Cannot find pyCGNS config ?"
    sys.exit(1)

#import CGNS
import CGNS.WRA._mll    as Mll 
import CGNS.WRA.wrapper as W
import numpy            as Num

import time

class xFAILED(Exception):
  def __init__(self):
    print "#### *** FAILED ON LAST TEST ***"

TDBNAME ="./adf-1.cgns"
TDBNAME2="./adf-2.cgns"
TDBNAME3="./adf-3.cgns"
TDBNAME4="./adf-4.cgns"

def tt(s,n=TDBNAME):
  import posixpath
  import os
  if posixpath.exists(n): os.unlink(n)
  print "#---#"
  print s

def ee(a,code=-1):
  if (a.error != -1):
    if (code == -1) or (a.error != code):
      print "#   # Error [%d][%s]"%(a.error,a.error_message(a.error))
      raise xFAILED
    else:
      print "#   # Got expected error [%s]"%a.error_message(a.error)

_time_start=0      
def start():
  global _time_start
  _time_start=time.time()

def stop():
  _time_stop=time.time()
  print "#   # time %.2fs"%(_time_stop-_time_start)
  
# -------------------------------------------------------------------------
# array defs
ar1=Num.zeros((3,7),'f')
ar2=Num.zeros((3,7),'d')
ar3=Num.reshape(Num.arange(189.0),(3,7,9))#,order='F')
ar4=Num.zeros((3,1,1,2,3,1,2),'i')
ar5=Num.ones ((2,),'i')
ar6=Num.ones ((2,),'i')*3
ar7=Num.array(["This is a string testing now char arrays"])
ar8=Num.reshape(Num.arange(189.0),(3,7,9),order='F')

# -------------------------------------------------------------------------
# EXTENSIVE TEST of ADF wrapper (adf then of ADF lib... :)
print "-" *70
print """Tests the implementation of Python wrapper on top of ADF libraries.
The test(s) with (???) marks have not the behavior I expected they could
have. This can be [1] bad implementation on my side, [2] misunderstanding
of the ADF lib behavior, [3] problem with ADF lib implementation."""

# -------------------------------------------------------------------------
print "-" *70
print "#   # Database-level routine"

# -------------------------------------------------------------------------
tt("# 01# open/close a database")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
a.database_close()
del a

# -------------------------------------------------------------------------
tt("# 02# check args number at open call")
try:
 a=W.pyADF(TDBNAME,"OPEN_WRITE")
except TypeError: pass

# -------------------------------------------------------------------------
try:
  tt("# 03# check status at open call")
  a=W.pyADF(TDBNAME,"OPEN",Mll.adf.NATIVE)
except:
  pass

# -------------------------------------------------------------------------
try:
  tt("# 04# check format at open call")
  a=W.pyADF(TDBNAME,Mll.adf.NEW,"IEEE_BUG")
  ee(a)
except:
  pass

# -------------------------------------------------------------------------
tt("# 05# get the format")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
print "#   # format: ", a.database_get_format()
del a

# -------------------------------------------------------------------------
tt("# 06# get root node id")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
print "#   # root ",a.root()
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
try:
  tt("# 07# empty name at open call -> succeed (???)")
  a=W.pyADF("",Mll.adf.NEW,Mll.adf.NATIVE)    
except:
  pass

# -------------------------------------------------------------------------
tt("# 08# file does not exist, try to read")
try:
 a=W.pyADF(TDBNAME,Mll.adf.READ_ONLY,Mll.adf.NATIVE)
except : pass

# -------------------------------------------------------------------------
tt("# 09# two NEW opens on the same file ")
try:
  a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
  ee(a)
  b=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
except: 
  a.database_close()
  del a

# -------------------------------------------------------------------------
tt("# 10# enumerates checks")
a=[]
a+=[Mll.adf.NATIVE]
a+=[Mll.adf.IEEE_BIG]
a+=[Mll.adf.IEEE_LITTLE]
a+=[Mll.adf.CRAY]
print "#   # enums: ",a
print "#   # enums: ",Mll.adf.ADF_OPENFORMAT
print "#   # enums: ",CGNS.midlevel.adf.ADF_OPENFORMAT
a=[]
a+=[Mll.adf.ADF_OPENFORMAT[Mll.adf.NATIVE]]
a+=[Mll.adf.ADF_OPENFORMAT[Mll.adf.IEEE_BIG]]
a+=[Mll.adf.ADF_OPENFORMAT[Mll.adf.IEEE_LITTLE]]
a+=[Mll.adf.ADF_OPENFORMAT[Mll.adf.CRAY]]
print "#   # reverse enums: ",a


# -------------------------------------------------------------------------
print "-" *70
print "#   # Data structure and management routines"

# -------------------------------------------------------------------------
tt("# 20# Create a node")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
print "#   # child ",a.str(id)
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 21# Create two nodes with same name")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.rootid,"childOne")
print "#   # child ",a.str(id)
ee(a)
id=a.create(a.rootid,"childOne")
ee(a,26)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 22# Get node id")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
id2=a.get_node_id(a.root(),"TOP")
if (     (a.get_name(id) != a.get_name(id2))
     or  (a.get_label(id) != a.get_label(id2))
     or  (a.get_data_type(id) != a.get_data_type(id2))):
  raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 23# Delete a node")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
a.delete(a.root(),id)
ee(a)
a.get_node_id(a.root(),"TOP")
ee(a,29)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 24# Get/Put node name")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
print "#   # name ",a.get_name(id)
ee(a)
a.put_name(a.root(),id,"ZAP")
ee(a)
print "#   # name ",a.get_name(id)
ee(a)
if (a.get_name(id) != "ZAP"):
  raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 25# Get number of (empty) children")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
n=a.number_of_children(id)
print "#   # children ",n
ee(a)
if (n != 0):  raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
import random
def randomTree(node,depth,a):
  if depth:
    for n in range(1,random.randrange(3,6)):
      p="[%.2d]"%n
      #print (10-depth)*"    ",p
      id=a.create(node,p)
      ee(a)
      randomTree(id,depth-1,a)
             
# -------------------------------------------------------------------------
tt("# 26# Create a random tree",TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
depth=5
start()
randomTree(id,depth,a)
stop()
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 27# Children list")
a=W.pyADF(TDBNAME2,Mll.adf.READ_ONLY,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
n=a.number_of_children(id)
ee(a)
print "#   # children ",n
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
count=0
def parseSub(node,a,decay):
  decay+=1
  for n in a.children_names(node):
#    print "    "*decay,n
    global count
    count+=1
    id=a.get_node_id(node,n)
    parseSub(id,a,decay)
  decay-=1    
                     
# -------------------------------------------------------------------------
tt("# 28# Parse previous random tree")
a=W.pyADF(TDBNAME2,Mll.adf.READ_ONLY,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
start()
parseSub(id,a,0)
stop()
ee(a)
print "#   # number of nodes ",count
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 29# Delete sub-tree of previous random tree")
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"/TOP/[01]/[01]")
ee(a)
a.delete(a.root(),id) # BAD FATHER RETURNS AN ADF ERROR [24]
ee(a,24)
#ee(a,29)
pid=a.get_node_id(a.root(),"/TOP/[01]")
ee(a)
a.delete(pid,id)
ee(a,24)
a.database_close()
ee(a)
del a # THE A CLASS INSTANCE HAS TO BE DELETED
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
count=0
parseSub(a.root(),a,0)
print "#   # number of nodes ",count
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 30# Bad node name (contains /)")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
id=a.create(id,"/TOP-SON")
ee(a,56)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 31# Bad node name (string too long)")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
id=a.create(id,"X"*33)
ee(a,4)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
def completeTree(node,depth,nchildren,a):
  if depth:
    for n in range(1,nchildren+1):
      p="[%.2d]"%n
      #print (10-depth)*"    ",p
      id=a.create(node,p)
      ee(a)
      completeTree(id,depth-1,nchildren,a)
             
# -------------------------------------------------------------------------
tt("# 32# Creates a complete tree node=5^depth=4",TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
start()
completeTree(id,4,5,a)
stop()
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 33# Parse time for previous complete tree")
a=W.pyADF(TDBNAME2,Mll.adf.READ_ONLY,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
count=0
start()
parseSub(id,a,0)
stop()
ee(a)
print "#   # number of nodes ",count
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 34# Move a child (change its name)")
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"/TOP/[01]/[02]/[03]/[04]")
ee(a)
pid=a.get_node_id(a.root(),"/TOP/[01]/[02]/[03]")
ee(a)
nid=a.get_node_id(a.root(),"/TOP/[03]/[02]")
ee(a)
a.put_name(pid,id,"MOVED")
ee(a)
a.move_child(pid,id,nid)
ee(a)
count=0
id=a.get_node_id(a.root(),"/TOP")
print "#   # re-parse "
start()
parseSub(id,a,0) # recount a get parse time not-really interesting :(
stop()
ee(a)
print "#   # number of nodes ",count
ee(a)
id=a.get_node_id(a.root(),"/TOP/[03]/[02]/MOVED") # check path
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 40# Create a db for link test (re-create complete tree base)",TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
completeTree(id,4,5,a)
ee(a)
a.database_close()
ee(a)
del a
tt("#   # Actually create a link ",TDBNAME3)
a=W.pyADF(TDBNAME3,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"SUB-TOP")
ee(a)
a.link(id,"LINK-TO-ELSEWHERE",TDBNAME2,"/TOP/[01]/[02]")
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
print "-" *70
print "#   # Node attributes routines"
tt("# 50# Get the node label")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
print "#   # [ Root Node of ADF File ]=[", a.get_label(a.root()), "]"
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
print "-" *70
tt("# 51# Get/Set the node label")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
a.get_label(id)
ee(a)
l1="NEW-NODE-LABEL"
a.set_label(id,l1)
ee(a)
l2=a.get_label(id)
ee(a)
print "#   # [ NEW-NODE-LABEL ]=[", a.get_label(id), "]"
if (l2 != l2): raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
print "-" *70
tt("# 52# Get the node data type")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a) 
print "#   # MT = ", a.get_data_type(id)
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 53# Get/Set the node data type")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
d1="R8"
a.put_dimension_information(id,d1,(1,2,3))
ee(a)
d2=a.get_data_type(id)
ee(a)
if (d1 != d2): raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 54# Bad data type")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
d1="R8-3C&"
a.put_dimension_information(id,d1,(1,2,3))
ee(a,31)
del a

# -------------------------------------------------------------------------
tt("# 55# Try to get dimensions")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
try:
  d1=a.get_dimension_values(id)
except Mll.adf.error:
  ee(a,27)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 56# Change dimension")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
d1=(1,2,3)
a.put_dimension_information(id,"R4",d1)
d2=a.get_dimension_values(id)
print a.error
ee(a)
print "#    # (1,2,3) =",d2
if (d1 != d2): raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 57# Dimensions number")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
id=a.create(a.root(),"TOP")
ee(a)
a.put_dimension_information(id,"R4",(1,2,3,4,5))
ee(a,31)
a.put_dimension_information(id,"I4",(1,2,3,4,5))
ee(a)
d=a.get_dimension_values(id)
ee(a)
print "#   # 5 =", len(d)
print "#   # (1,2,3,4,5) =", d
if (len(d) != 5): raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
print "-" *70
print "#   # Data I/O routines"
tt("# 60# Add data",n=TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
a.put_dimension_information(id,"R8",ar3.shape)
ee(a)
a.write_all_data(id,ar3)
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 61# Read data")
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
ar0=a.read_all_data(id)
ee(a)
d=a.get_dimension_values(id)
ee(a)
arc=Num.asarray(ar0,order='C')
print "#   # read dim        :",d
print "#   # ref  table shape:",ar3.shape
print "#   # read table shape:",ar0.shape
print "#   # C    table shape:",arc.shape
print "#   # PY/C fortran    :",Num.isfortran(ar3)
print "#   # ADF/F fortran   :",Num.isfortran(ar0)
print "#   # ADF/F/C fortran :",Num.isfortran(arc)
print "#   # PY/C            :",ar3[2][3][4]
print "#   # ADF/F           :",ar0[2][3][4]
print "#   # ADF/F/C         :",arc[2][3][4]
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 62# Add data I4",n=TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
a.put_dimension_information(id,"I4",ar5.shape)
ee(a)
a.write_all_data(id,ar5)
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 63# Read data I4")
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
ar6=a.read_all_data(id)
ee(a)
print "#   # ",ar6[0]," = ",ar5[0]
print "#   # ",ar6[1]," = ",ar5[1]
if (ar6[0] != ar6[0]): raise xFAILED
if (ar6[1] != ar6[0]): raise xFAILED
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 64# Add data C1",n=TDBNAME2)
a=W.pyADF(TDBNAME2,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
print "#   # ",ar7.tostring()
print "#   # ",ar7.shape
print "#   # ",len(ar7[0])
a.put_dimension_information(id,"C1",(len(ar7[0]),))
ee(a)
a.write_all_data(id,ar7)
ee(a)
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 65# Read data C1")
a=W.pyADF(TDBNAME2,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.get_node_id(a.root(),"TOP")
ee(a)
ar7=a.read_all_data(id)
ee(a)
print "#   # ",ar7.tostring()
print "#   # ",ar7.shape
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 66# Write/Read data R4",TDBNAME4)
a=W.pyADF(TDBNAME4,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
sid=a.create(id,"DATA")
ee(a)
a.put_dimension_information(sid,"R4",(3,2,7))
ar=Num.ones((3,2,7),dtype=Num.float32,order='F') # stands for FORTRAN
a.write_all_data(sid,ar)
ee(a)
a.database_close()
del ar

a=W.pyADF(TDBNAME4,Mll.adf.OLD,Mll.adf.NATIVE)
id1=a.get_node_id(a.root(),"TOP")
ee(a)
id2=a.get_node_id(id1,"DATA")
ee(a)
ar99=a.read_all_data(id2)
ee(a)
print "#   # float32 ==",ar99.dtype.name
print "#   # (3,2,7) ==",ar99.shape
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
tt("# 67# Fortran array")
a=W.pyADF(TDBNAME4,Mll.adf.OLD,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP-F")
ee(a)
print "#   # (3,7,9) == ",ar8.shape
a.put_dimension_information(id,"R8",ar8.shape)
ee(a)
a.write_all_data(id,ar8)
ee(a)
a.database_close()
ee(a)
del a
a=W.pyADF(TDBNAME4,Mll.adf.OLD,Mll.adf.NATIVE)
id=a.get_node_id(a.root(),"TOP-F")
ee(a)
ar=a.read_all_data(id)
ee(a)
arc=Num.asarray(ar,order='C')
print "#   # ADF/F    ",Num.isfortran(ar)
print "#   # PY/F     ",Num.isfortran(ar8)
print "#   # ADF/PY/C ",Num.isfortran(arc)
print "#   # ADF/F    ",ar[2][1][5]
print "#   # PY/F     ",ar8[2][1][5]
print "#   # ADF/PY/C ",arc[2][1][5]
a.database_close()
ee(a)
del a

# -------------------------------------------------------------------------
print "-" *70
print "#   # Miscellaneous routines"
tt("# 90# Flush")
a=W.pyADF(TDBNAME,Mll.adf.NEW,Mll.adf.NATIVE)
ee(a)
id=a.create(a.root(),"TOP")
ee(a)
a.flush_to_disk(id)
ee(a)
a.database_close()
ee(a)
del a

print "-" *70
print "End test suite"

# -------------------------------------------------------------------------
# last line
