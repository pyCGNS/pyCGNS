#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
import CGNS.errors    as ERR
import CGNS.WRA._mll  as __MLL
import CGNS.WRA._adf  as __ADF

import posixpath
import copy

def wpdbg(msg):
  if 0: print "#wrap# %s"%msg

__doc__="""
The pyCGNS wrap classes are simple wrappers on top of the ADF and the MLL
libraries.
"""

# ======================================================================
class pyADF:
    """
    The *pyADF* class holds a connection to a *CGNS* file provides the user
    with a *Python*-like interface to *ADF* calls. 

    Each method or attribute use actually runs the ADF library functions
    
    ADF calls to DB have the same name without the ```ADF``` prefix and with
    a lowercase name. Argument list always try to be as close as possible
    to the ADF original argument list
    """
    (UNKNOWN,OPEN,CLOSE,GHOST)=range(4)
    # --------------------------------------------------
    def __init__(self,name,status,format):
      """
      Opens a CGNS database, returns the instance required for subsequent
      calls on this database. To close the database, use the Python del
      statement or the 'database_close' method.

      Arguments:

        - 'name-of-file:String'

        - 'open-mode:ADF_OPENSTATUS'

        - 'file-format:ADF_OPENFORMAT)'

      Return:

        - a pyADF instance
      """
      import CGNS.WRA._adf  as __ADF
      self.__state=self.UNKNOWN
      if (type(name) != type("")) and not status and not format:
        # open here from a root-id
        self.__adf=__ADF.database_open(name)
        self.__state=self.OPEN
      elif not name:
        raise ERR.CGNS_NoFileName
      elif (status not in __ADF.ADF_OPENSTATUS.values()):
        raise ERR.CGNS_BadADFstatus
      elif (format not in __ADF.ADF_OPENFORMAT.values()):
        raise ERR.CGNS_BadADFformat
      else:
        self.__adf=__ADF.database_open(name,status,format)
        if (self.__adf.error != -1):
          self.__state=self.OPEN
      self.__name=name
    # --------------------------------------------------
    def database_close(self):
      """
      Closes a CGNS database. It does `not' delete the object, so we suggest
      you use the Python del statement instead of this method.
      """
      return self.__adf.database_close()
    # --------------------------------------------------
    def database_delete(self,filename):
      """
      Not implemented at ADF library level.
      """
      wpdbg("database_delete")
      return self.__adf.database_delete(filename)
    # --------------------------------------------------
    # if rootid is 0, take connection root id
    def database_get_format(self,rootid=0):
      wpdbg("database_get_format")
      if not rootid:
        r=self.__adf.root()
      else:
        r=rootid
      return self.__adf.database_get_format(r)
    # --------------------------------------------------
    # if rootid is 0, take connection root id
    def database_set_format(self,rootid,format):
      wpdbg("database_set_format")
      if (format not in __ADF.ADF_OPENFORMAT.values()):
        raise ADF_X_NOSUCHOPENFORMAT
      if not rootid:
        r=self.__adf.root()
      else:
        r=rootid
      return self.__adf.database_set_format(r,format)
    # --------------------------------------------------
    def database_garbage_collection(self,node):
      return self.__adf.database_garbage_collection(node)
    # --------------------------------------------------
    def database_version(self,rootid=0):
      if not rootid:
        r=self.__adf.root()
      else:
        r=rootid
      return self.__adf.database_version(r)
    # --------------------------------------------------
    def library_version(self):
      return self.__adf.library_version()
    # --------------------------------------------------
    def get_root_id(self,node):
      wpdbg("get_root_id")
      return self.__adf.get_root_id(node)
    # --------------------------------------------------
    def create(self,parent,name):
      wpdbg("create")
      return self.__adf.create(parent,name)
    # --------------------------------------------------
    def delete(self,parent,node):
      wpdbg("delete")
      return self.__adf.delete(parent,node)
    # --------------------------------------------------
    def put_name(self,parent,node,name):
      wpdbg("put_name")
      return self.__adf.put_name(parent,node,name)
    # --------------------------------------------------
    def get_name(self,node):
      wpdbg("get_name")
      return self.__adf.get_name(node)
    # --------------------------------------------------
    def number_of_children(self,node):
      wpdbg("number_of_children")
      return self.__adf.number_of_children(node)
    # --------------------------------------------------
    def children_names(self,node):
      wpdbg("children_names")
      return self.__adf.children_names(node)
    # --------------------------------------------------
    def move_child(self,parent,node,newparent):
      wpdbg("move_child")
      return self.__adf.move_child(parent,node,newparent)
    # --------------------------------------------------
    def get_node_id(self,parent,name):
      wpdbg("get_node_id")
      return self.__adf.get_node_id(parent,name)
    # --------------------------------------------------
    def get_label(self,node):
      wpdbg("get_label")
      return self.__adf.get_label(node)
    # --------------------------------------------------
    def set_label(self,node,label):
      wpdbg("set_label")
      return self.__adf.set_label(node,label)
    # --------------------------------------------------
    def get_data_type(self,node):
      wpdbg("get_data_type")
      return self.__adf.get_data_type(node)
    # --------------------------------------------------
    def get_number_of_dimensions(self,node):
      wpdbg("get_number_of_dimensions")
      return self.__adf.get_number_of_dimensions(node)
    # --------------------------------------------------
    def get_dimension_values(self,node):
      wpdbg("get_dimension_values")
      return self.__adf.get_dimension_values(node)
    # --------------------------------------------------
    # datatype is a string
    # dims is a tuple of integers
    def put_dimension_information(self,node,datatype,dims):
      return self.__adf.put_dimension_information(node,datatype,dims)
    # --------------------------------------------------
    def is_link(self,node):
      return self.__adf.is_link(node)
    # --------------------------------------------------
    def link(self,parent,name,destfile,destnode):
      return self.__adf.link(parent,name,destfile,destnode)
    # --------------------------------------------------
    def get_link_path(self,node):
      return self.__adf.get_link_path(node)
    # --------------------------------------------------
    def read_data(self):
      return self.__adf.read_data()
    # --------------------------------------------------
    def read_all_data(self,node):
      r=self.__adf.read_all_data(node)
      return r
    # --------------------------------------------------
    def write_data(self):
      return self.__adf.write_data()
    # --------------------------------------------------
    def write_all_data(self,node,array):
      return self.__adf.write_all_data(node,array)
    # --------------------------------------------------
    def error_message(self,code):
      return self.__adf.error_message(code)
    # --------------------------------------------------
    def set_error_state(self,state):
      return self.__adf.set_error_state(state)
    # --------------------------------------------------
    def get_error_state(self):
      return self.__adf.get_error_state()
    # --------------------------------------------------
    def flush_to_disk(self,node):
      return self.__adf.flush_to_disk(node)

    # --------------------------------------------------
    # Extra methods
    # --------------------------------------------------    
    def root(self):
      wpdbg("x root")
      return self.__adf.root()
    # --------------------------------------------------
    def children(self,node=0):
      wpdbg("x children")
      if (node==0):
        return self.__adf.children(self.__adf.root())
      else:
        return self.__adf.children(node)
    # --------------------------------------------------
    # this should be called when you try to put an id into
    # a string. Or you can take the format as example...
    def str(self,nodeid):
      return "%.48g"%nodeid
    # --------------------------------------------------
    # we do NOT follow link
    def nodeAsDict(self,node):
      #print 'start',70*'-'
      d={}
      d['id']=node
      d['name']=self.get_name(node)
      #print "nodeAsDict :", d['name']
      if self.is_link(node):
        (fx,nx)=self.get_link_path(node)
        d['datatype']='LK' 
        d['label']=''
        d['file']=fx
        d['path']=nx
        d['children']=[]
        d['dimensions']=()
      else:
        d['datatype']=self.get_data_type(node)
        if (self.get_number_of_dimensions(node)):
          d['dimensions']=self.get_dimension_values(node)
        else:
          d['dimensions']=()
        d['label']=self.get_label(node)
        d['children']=self.children_names(node)
      #print "nodeAsDict :", d['children']
      #print 'end',70*'-'
      return d
    # --------------------------------------------------    
    def __nonzero__(self):
      return True
    # --------------------------------------------------    
    def __ne__(self,value):
      return not self.__eq__(value)
    # --------------------------------------------------    
    def __repr__(self):
      return '<pyADF instance from "%s">'%self.__name
    # --------------------------------------------------    
    def __eq__(self,value):
      return (    (self.__class__ == value.__class__)
              and (self.__dict__ == value.__dict__))
    # --------------------------------------------------    
    def __getattr__(self,name):
      """
      Purpose:
      
      Returns the allowed attributes on this object.

      Attributes:

        - 'rootid:Double', the ADF root node id of the current database

        - 'error:Integer', the last ADF error code
      """
      if (name == 'rootid'): return self.__adf.root()
      if (name == 'error') : return self.__adf.error
      return self.__dict__[name]
    
# ======================================================================
class pyCGNS:
    """
    pyCGNS: wrapper to CGNS calls
    - pyCGNS creation opens a database
    - CGNS calls are partly implemented
    """
    # --------------------------------------------------
    def __init__(self,name,mode=-9):
      """-Creates a CGNS database -(pyCGNS)

         ''*pyCGNS*'(file-name:S,file-mode:I)'

         The file mode is an enumerate. It can have the values:
         MODE_READ, MODE_WRITE, MODE_MODIFY
      """
      import CGNS.WRA._mll  as __MLL
      self.__alive=0
      self.__name=name
      if (mode==-9): mode=_MLL_.MODE_READ
      if (type(mode)==type(1)):
        if (mode not in _MLL_.OpenMode_.keys()): raise ERR.CGNS_BadOpenMode
        self.__mode=mode
        self.__modestring=_MLL_.OpenMode_[mode]
      else:
        if (mode not in _MLL_.OpenMode.keys()): raise ERR.CGNS_BadOpenMode
        self.__mode=_MLL_.OpenMode[mode]
        self.__modestring=mode
      self.__lastPath=[]
      try:
        self.__db=__MLL.connect(self.__name,self.__mode)
      except:
        if (self.__mode==3): # v2/v3 issue with MODE_MODIFY enum
          self.__mode=2
          self.__db=__MLL.connect(self.__name,self.__mode)
      self.__alive=1      
    # --------------------------------------------------
    def close(self):
      """-Close the current CGNS file -(pyCGNS)

         'None='*close*'()'

         Close is performed by *del* if not already (explicitely) done.
      """
      if (self.__alive):
          self.__db.close()
          self.__alive=0
    # --------------------------------------------------
    def __del__(self):
      if (self.__alive):        
          self.__db.close()
          self.__alive=0
    # --------------------------------------------------
    # the second arg is set flag
    # 1 means set the current node
    # 0 means do NOT set the current node
    # Either calls are returning the ADF node id
    def goto(self,baseidx,path=()):
      """-Set the current node -Node

         'node-id:D='*goto*'(base-id:I,path:((node-type:S,node-id:I),...))'

         The 'goto' sets the current node to the leaf of the given path.
         The path itself is a list of tuples. Each tuple contains the
         node type a first argument, its index as second arg.
         Note the returned id is not (yet) trustable.
      """
      self.__db.goto(baseidx,1,path)
      self.__lastPath=copy.copy(path) # save it for userdata for example
    # --------------------------------------------------
    def deletenode(self,name):
      """-Delete the given node -Node

         'None='*delenode*'(name:S)'

         Removes the current node (and its children).
      """
      return self.__db.deleteNode(name)
    # --------------------------------------------------
    def id(self,baseidx,path):
      """-Get the ADF id of a given node -(pyCGNS)

         'node-id:D='*id*'(base-id:I,path:((node-type:S,node-id:I),...))'

         Uses the same syntax as 'goto', but doesn't set the current node.
         This id call is not trustable, it has not beeen tested and probably
         has a destructive effect on cgnslib global variables...
         See 'goto' remarks
      """
      return self.__db.goto(baseidx,0,path)
    # --------------------------------------------------
    def basewrite(self,name,cdim,pdim):
      """-Create a new base in an existing CGNS file -Base

         'base-idx:I='*basewrite*'(base-name:S,cell-dim:I,phys-dim:I)'

          Args are base name, cell and physical dimensions.
          Returns the new id of the created base
      """
      return self.__db.baseWrite(name,cdim,pdim)
    # --------------------------------------------------
    def baseread(self,id):
       """-Get infos about a given base -Base
         
         '(base-idx:I,base-name:S,cell-dim:I,phys-dim:I)='*baseread*'(base-idx:I)'
         
          Arg is base id, returns a tuple with base information.
       """
       return self.__db.baseRead(id)
    # --------------------------------------------------
    def baseid(self,id):
      """-Get the ADF id of a base -undocumented MLL

         'base-id:D='*baseid*'(base-id:I)'

         The argument is the MLL id, the return value is the ADF id, it
         is a double float value. Such an id cannot be obtained if the
         CGNS/ADF file has been open as write only.
      """
      if self.__modestring not in [_MLL_.MODE_READ,_MLL_.MODE_MODIFY]:
          raise ERR.CGNS_FileShouldBeReadable
      return self.__db.baseId(id)
    # --------------------------------------------------
    def linkread(self):
      """-Get infos about a given link -Link

         '(file-name:S,target-name:S))='*linkread*'()'

         Uses the current node as argument node previously set by
         'goto' call.
      """
      return self.__db.linkRead()
    # --------------------------------------------------
    def linkwrite(self,sourcenode,destfile,destnode):
      """-Create a link -Link

         'None='*linkwrite*'(source-name:S,file-name:S,target-name:S)'

         Args are the link name, the destination file, destination node
         name. Returns None.
      """
      return self.__db.linkWrite(sourcenode,destfile,destnode)
    # --------------------------------------------------
    def islink(self): 
      """-Test if current node is a link -Link

         'true-if-is-link:I='*islink*'()'

         Uses current node.
      """
      return self.__db.isLink()
    # --------------------------------------------------
    def descriptorread(self,index):
      """-Get the descriptor contents -Descriptor

         '(desc-name:S,desc-text:S)='*descriptorread*'(desc-id:I)'

         The current node is used.
      """
      return self.__db.descriptorRead(index)
    # --------------------------------------------------
    def descriptorwrite(self,nodename,text):
      """-Create or update a descriptor under the current node -Descriptor

         'None='*descriptorwrite*'(desc-name:S,desc-test:S)'

         Unfair-remark: We should get the descriptor id as returned argument.
      """
      return self.__db.descriptorWrite(nodename,text)
    # --------------------------------------------------
    def nzones(self,id):
      """-Count zones in the base -Zone

         'number-of-zones:I='*nzones*'(base-id:I)'

         The zone count is a max, the zone ids are starting from 1 (one). 
         Thus, using 'nzones' with a 'range' function
         should be done with 'range(1,file.nzones(base)+1)'
      """
      return self.__db.nzones(id)
    # --------------------------------------------------
    def zoneid(self,bid,zid):
      """-Get the ADF id of a zone -undocumented MLL

         'zone-id:D='*zoneid*'(base-id:I,zone-id:I)'

         See 'baseid' remarks.
      """
      if self.__modestring not in [_MLL_.MODE_READ,_MLL_.MODE_MODIFY]:
          raise ERR.CGNS_FileShouldBeReadable
      return self.__db.zoneId(bid,zid)
    # --------------------------------------------------
    def zoneread(self,bid,zid):
      """-Get infos about a given zone -Zone
      
         '(base-id:I,zone-id:I,zone-name:S,size-tuple(I,...)='*zoneread*'(base-id:I,zone-id:I)'

         The tuple returns a useful informatinos, including arguments ids.
         The dimension tuple size depends on the zone size. To get this
         tuple size, use the 'len' function.
      """
      return self.__db.zoneRead(bid,zid)
    # --------------------------------------------------
    def zonetype(self,bid,zid):
      """-Get the type of a given zone -Zone

         'zone-type:S='*zonetype*'(base-id:I,zone-id:I)'

         The returned string can be used as entry key into ZoneType
         dictionnary, in order to get the actual integer value for the
         corresponding enumerate.
      """
      return self.__db.zoneType(bid,zid)
    # --------------------------------------------------
    def zonewrite(self,baseidx,name,szlist,zonetype):
      """-Create a new zone -Zone

         'zone-id:I='*zonewrite*'(base-id:I,zone-name:S,size-tuple:(I,...),zone-type:S)'

         See 'zonetype' remarks.
      """
      if zonetype not in _MLL_.ZoneType: raise ERR.CGNS_BadZoneType
      # bsize=self.baseread(baseidx) # cannot be called if open for wrtie...
      bsize=3
      if (len(szlist)==6): szlist=szlist+(0,0,0)
      if (type(zonetype)==type("")): zonetype=_MLL_.ZoneType[zonetype]
      print baseidx,name,tuple(szlist),zonetype
      sizelist=self.checkzonesize(bsize,szlist,zonetype)
      return self.__db.zoneWrite(baseidx,name,tuple(szlist),zonetype)
    # --------------------------------------------------
    def equationsetchemistryread(self):
      """-Read chemistry flags -Flow Equation Set

         'equation-flags:(I,I)='*equationsetchemistryread*'()'

         No Comment
      """
      return self.__db.equationsetChemistryRead()
    # --------------------------------------------------
    def equationsetread(self):
      """-Get equation set info -Flow Equation Set

         'equation-dim:(I,I,I,I,I)='*equationsetread*'()'

         No Comment
      """
      return self.__db.equationsetRead()
    # --------------------------------------------------
    def equationsetwrite(self,d):
      """-Set equation set info -Flow Equation Set

         'None='*equationsetwrite*'(equation-dim:I)'

         No Comment
      """
      return self.__db.equationsetWrite(d)
    # --------------------------------------------------
    def governingread(self):
      """-Get governing equations info -Flow Equation Set

         'governing-eq-type:I='*governingread*'()'

         No Comment
      """
      return self.__db.governingRead()
    # --------------------------------------------------
    def governingwrite(self,d):
      """-Set governing equations info -Flow Equation Set

         'None='*governingwrite*'(governing-eq-type:I)'

         No Comment
      """
      if d not in _MLL_.GoverningEquationsType.keys():
        raise ERR.CGNS_BadGoverningType
      return self.__db.governingWrite(_MLL_.GoverningEquationsType[d])
    # --------------------------------------------------
    def diffusionread(self):
      """-Get diffusion info -Flow Equation Set

         '(I,I,I,I,I)='*diffusionread*'()'

         No Comment
      """
      return self.__db.diffusionRead()
    # --------------------------------------------------
    def diffusionwrite(self,d):
      """-Set diffusion info -Flow Equation Set

         'None='*diffusionwrite*'(I,I,I,I,I)'

         No Comment
      """
      return self.__db.diffusionWrite(d)
    # --------------------------------------------------
    def modelread(self,label):
      """-Get model info -Flow Equation Set

         '(model-name:S,model-type:I)='*modelread*'(nodel-name:S)'

         No Comment
      """
      return self.__db.modelRead(label)
    # --------------------------------------------------
    def modelwrite(self,label,mt):
      """-Set model info -Flow Equation Set

         'None='*modelwrite*'(model-name:S,model-type:I)'

         No Comment
      """
      if mt not in _MLL_.ModelType.keys(): raise ERR.CGNS_BadModelType
      return self.__db.modelWrite(label,_MLL_.ModelType[mt])
    # --------------------------------------------------
    def stateread(self):
      """-Get state info -Flow Equation Set

         'state-description:S='*stateread*'()'

         No Comment
      """
      return self.__db.stateRead()
    # --------------------------------------------------
    def statewrite(self,name):
      """-Set state info -Flow Equation Set

         'None='*statewrite*'(state-description:S)'

         No Comment
      """
      return self.__db.stateWrite(name)
    # --------------------------------------------------
    def simulationtyperead(self,bid):
      """-Get simulation type info -Flow Equation Set

         'simulation-type:I='*simulationtyperead*'(base-id:I)'

         No Comment
      """
      return self.__db.simulationTypeRead(bid)
    # --------------------------------------------------
    def simulationtypewrite(self,bid,simtype):
      """-Set simulation type info -Flow Equation Set

         'None='*simulationtypewrite*'(base-id:I,simulation-type:I)'

         No Comment
      """
      if simtype not in _MLL_.SimulationType.keys():
        raise ERR.CGNS_BadSimulationType
      return self.__db.simulationTypeWrite(bid,_MLL_.SimulationType[simtype])
    # --------------------------------------------------
    def rotatingread(self):
      """-Get the rotation parameters -Rotating Coordinates

         '(rate-vector:(D,...),center:(D,...))='*rotatingread*'()'

         The '(D,...)' have the base physical dimension (i.e. 2 in
         2D and 3 in 3d).
      """
      return self.__db.rotatingRead()
    # --------------------------------------------------
    def rotatingwrite(self,rv,rc):
      """-Set the rotation parameters -Rotating Coordinates

         'None='*rotatingwrite*'(rate-vector:(D,...),center:(D,...))'

         See 'rotatingread'
      """
      return self.__db.rotatingWrite(rv,rc)
    # --------------------------------------------------
    def axisymread(self,bid):
      """-Get the axisymmetry parameters -Axisymmetry

         '(reference-point:(D,D),axis-vector:(D,D))='*axisymread*'(base-id:I)'

         Should be 2D.
      """
      return self.__db.axisymRead(bid)
    # --------------------------------------------------
    def axisymwrite(self,bid,rv,rc):
      """-Set the axisymmetry parameters -Axisymmetry

         'None='*axisymwrite*'(base-id:I,reference-point:(D,D),axis-vector:(D,D))'

         Should be 2D.
      """
      return self.__db.axisymWrite(bid,rv,rc)
    # --------------------------------------------------
    def gravityread(self,bid):
      """-Get the gravity vector -Auxiliary Data

         '(gravity-vector:(D,...))='*gravityread*'(base-id:I)'

         Size depends on physical dimension of base.
      """
      return self.__db.gravityRead(bid)
    # --------------------------------------------------
    def gravitywrite(self,bid,gv):
      """-Set the gravity vector -Auxiliary Data

         'None='*gravitywrite*'(base-id:I,gravity-vector:(D,...))'

         Size depends on physical dimension of base.
      """
      return self.__db.gravityWrite(bid,gv)
    # --------------------------------------------------
    def gridlocationread(self):
      """-Get the grid location info -Grid

         'grid-location:I='*gridlocationread*'()'

         Under current node
      """
      return self.__db.gridlocationRead()
    # --------------------------------------------------
    def gridlocationwrite(self,gloc):
      """-Set the grid location info -Grid

         'None='*gridlocationwrite*'(grid-location:I)'

         Under current node
      """
      if gloc not in _MLL_.GridLocation.keys():
        raise ERR.CGNS_BadGridLocation
      return self.__db.gridlocationWrite(_MLL_.GridLocation[gloc])
    # --------------------------------------------------
    def nrigidmotions(self,bid,zid):
      """-Get number of rigid motion nodes -Rigid Grid Motion

         'number-of-motion:I='*nrigidmotions*'(base-id:I,zone-id:I)'

         No Comment
      """
      return self.__db.nRigidMotions(bid,zid)
    # --------------------------------------------------
    def narbitrarymotions(self,bid,zid):
      """-Get number of arbitrary motion nodes -Arbitrary Grid Motion

         'number-of-motion:I='*narbitrarymotions*'(base-id:I,zone-id:I)'

         No Comment
      """
      return self.__db.nArbitraryMotions(bid,zid)
    # --------------------------------------------------
    def rigidmotionread(self,bid,zid,mid):
      """-Get info about a rigid motion node -Rigid Grid Motion

         'return-tuple='*rigidmotionread*'(base-id:I,zone-id:I,motion-id:I)'

         The returned tuple contains: (name:S,RigidGridMotionType:I)
      """
      return self.__db.rigidMotionRead(bid,zid,mid)
    # --------------------------------------------------
    def arbitrarymotionread(self,bid,zid,mid):
      """-Get info about an arbitrary motion node -Arbitrary Grid Motion

         'return-tuple='*arbitrarymotionread*'(base-id:I,zone-id:I,motion-id:I)'

         The returned tuple contains: (name:S,ArbitraryGridMotionType:I)
      """
      return self.__db.arbitraryMotionRead(bid,zid,mid)
    # --------------------------------------------------
    def rigidmotionwrite(self,bid,zid,name,type):
      """-Create a new rigid motion node -Rigid Grid Motion

         'rigid-motion-id:I='*rigidmotionwrite*'(base-id:I,zone-id:I,name:S,type:I)'

         type:RigidGridMotionType
      """
      if type not in _MLL_.RigidGridMotionType.keys():
        raise ERR.CGNS_BadRigidGridMotionType
      return self.__db.rigidMotionWrite(bid,zid,name,_MLL_.RigidGridMotionType[type])
    # --------------------------------------------------
    def arbitrarymotionwrite(self,bid,zid,name,type):
      """-Create a new arbitrary motion node -Arbitrary Grid Motion

         'arbitrary-motion-id:I='*arbitrarymotionwrite*'(base-id:I,zone-id:I,name:S,type:I)'

         type:ArbitraryGridMotionType
      """
      if type not in _MLL_.ArbitraryGridMotionType.keys():
        raise ERR.CGNS_BadArbitraryGridMotionType
      return self.__db.arbitraryMotionWrite(bid,zid,name,_MLL_.ArbitraryGridMotionType[type])
    # --------------------------------------------------
    def nsols(self,bid,zid):
      """-Get count of solutions -Flow Solution

         'number-of-solutions:I='*nsols*'(base-id:I,zone-id:I)'

         No Comment
      """
      return self.__db.nsols(bid,zid)
    # --------------------------------------------------
    def solwrite(self,bid,zid,sname,glocation):
      """-Create a new solution -Flow Solution

         'sold-id:I='*solwrite*'(base-id:I,zone-id:I,sol-name:S,grid-location:S)'

         The grid location is a string. The corresponding enumerate can be
         found using the cross dictionnary. Examples:
         s=db.solwrite(1,z,mySolutionName,CGNS.CellCenter)
         s=db.solwrite(1,z,mySolutionName,CGNS.CellCenter_[myGridLocation])
      """
      if glocation not in _MLL_.GridLocation.keys():
        raise ERR.CGNS_BadGridLocation
      return self.__db.solWrite(bid,zid,sname,_MLL_.GridLocation[glocation])
    # --------------------------------------------------
    def solinfo(self,bid,zid,sid):
      """-Get infos about a given solution -Flow Solution

         '(grid-location:S,sol-name:S)='*solinfo*'(base-id:I,zone-id:I,sol-id:I)'

         See 'solwrite' remarks.
      """
      return self.__db.solInfo(bid,zid,sid)
    # --------------------------------------------------
    def solid(self,bid,zid,sid):
      """-Get the ADF id of a solution -undocumented MLL

         'sol-id:D='*solid*'(base-id:I,zone-id:I,sol-id:I)'

         See 'baseid' remarks.
      """
      if self.__modestring not in [_MLL_.MODE_READ,_MLL_.MODE_MODIFY]:
          raise ERR.CGNS_FileShouldBeReadable
      return self.__db.solId(bid,zid,sid)
    # --------------------------------------------------
    def ncoords(self,bid,zid):
      """-Count coordinates nodes in the zone -Coordinates

         'number-of-coords:I='*ncoords*'(base-id:I,zone-id:I)'

         See 'nzones' remarks.
      """
      return self.__db.ncoords(bid,zid)
    # --------------------------------------------------
    def coordread(self,bid,zid,cname,rmode=0):
      """-Read the coordinate array -Coordinates

         'coord-array:A='*coordread*'(base-id:I,zone-id:I,coord-name:S,read-mode:I)'

         The returned array is a 'pyArray' containing the data with the
         required format. Unfair-remark: a zone name is required, but all
         requests to nodes are done using integer ids.
         The read mode is default to 0, that is a C-like read (i,j,k).
         The mode=1 is fortran like read (k,j,i).
         Please, take care of the dimensions in that case, see the 'zoneread' remarks.
      """
      return self.__db.coordRead(bid,zid,cname,rmode)  
    # --------------------------------------------------
    def coordwrite(self,bid,zid,dtype,cname,darray):
      """-Create a coordinate array node -Coordinates

         'coord-id:I='*coordwrite*'(base-id:I,zone-id:I,data-type:S,node-name:S,data-array:A)'

         data-type:DataType
      """
      if dtype not in _MLL_.DataType.keys(): raise ERR.CGNS_BadDataType
      return self.__db.coordWrite(bid,zid,_MLL_.DataType[dtype],cname,darray)
    # --------------------------------------------------
    def coordinfo(self,bid,zid,cid):
      """-Get infos about a given coordinate -Coordinates

         '(data-type:S,node-name:S)='*coordinfo*'(base-id:I,zone-id:I,coord-id:I)'

         See 'coordwrite' remarks.
      """
      return self.__db.coordInfo(bid,zid,cid)
    # --------------------------------------------------
    def coordid(self,bid,zid,cid):
      """-Get the ADF id of a coordinate -undocumented MLL

         'coord-id:D='*coordid*'(base-id:I,zone-id:I,coord-id:I)'

         See 'baseid' remarks.
      """
      if self.__modestring not in [_MLL_.MODE_READ,_MLL_.MODE_MODIFY]:
          raise ERR.CGNS_FileShouldBeReadable
      return self.__db.coordId(bid,zid,cid)
    # --------------------------------------------------
    def nsections(self,bid,zid):
      """-Get lower range index -Element Connectivity

         'section-index:I='*nsections*'(base-id:I,zone-id:I)'

         The lower range is (imin, jmin, kmin)
      """
      return self.__db.nsections(bid,zid)
    # --------------------------------------------------
    def sectionread(self,bid,zid,sid):
      """-Get infos about a given section -Element Connectivity

         'return-tuple='*sectionread*'(base-id:I,zone-id:I,section-id:I)'

         Returns a tuple containing 'name:S' of the section, its 'type:I'
         'start:I' and 'end:I', 'last-bnd-index:I', 'parent-flag:I'.
      """
      return self.__db.sectionRead(bid,zid,sid)
    # --------------------------------------------------
    def sectionwrite(self,bid,zid,name,type,start,end,nb,ar):
      """-Write a section -Element Connectivity

         'section-id='*sectionwrite*'(base-id:I,zone-id:I,section-name:S,args...)'

         The trailing args are the 'type:I' of the section elements
         the 'start:I' and 'end:I' indices, 'last-bnd-index:I'
         index and at last the 'elements:A' array itself (of type 'type).
      """
      if type not in _MLL_.ElementType: raise ERR.CGNS_BadElementType
      return self.__db.sectionWrite(bid,zid,name,_MLL_.ElementType[type],
                                    start,end,nb,ar)
    # --------------------------------------------------
    def elementsread(self,bid,zid,sid):
      """-Get elements of a section -Element Connectivity

         '(elements:A,parents:A)='*elementsread*'(base-id:I,zone-id:I,section-id:I)'

         Returns two arrays of I
      """
      return self.__db.elementsRead(bid,zid,sid)
    # --------------------------------------------------
    def parentdatawrite(self,bid,zid,sid,ar):
      """-Write the parent data in a section -Element Connectivity

         'None='*parentdatawrite*'(base-id:I,zone-id:I,section-id:I,parent-data:A)'

         No return
      """
      return self.__db.parentDataWrite(bid,zid,sid,ar)
    # --------------------------------------------------
    def npe(self,t):
      """-Get the number of nodes for an element type -Element Connectivity

         'number:I='*npe*'(element-type:I)'

         element-type is an enumerate.
      """
      return self.__db.npe(t)
    # --------------------------------------------------
    def elementdatasize(self,bid,zid,sid):
      """-Get the number of elements for this section -Element Connectivity

         'number:I='*elementdatasize*'(base-id:I,zone-id:I,section-id:I)'

         See 'section-write'
      """
      return self.__db.elementDataSize(bid,zid,sid)
      # --------------------------------------------------
    def nfields(self,bid,zid,sid):
      """-Count fields in the solution -Flow Solution

         'number-of-fields:I='*nfields*'(base-id:I,zone-id:I,sol-id:I)'

         See 'nzones' remarks.
      """
      return self.__db.nfields(bid,zid,sid)
    # --------------------------------------------------
    def fieldread(self,bid,zid,sid,fname,dtype,imin,imax):
      """-Get the data array of a given solution field -Flow Solution

         'data-array:A='*fieldread*'(base-id:I,zone-id:I,sol-id:I,field-name:S,args...)'

         The trailing args are 'data-type:S' and tuples
         of indices: 'i-min:(I,I,I)' 'i-max:(I,I,I)'. These 'imin' and 'imax'
         tuples are forced to 3D, but only relevant
         values are used. Other values can be set to zero.
         Unfair-remark: field name is required.
      """
      if dtype not in _MLL_.DataType.keys(): raise ERR.CGNS_BadDataType
      return self.__db.fieldRead(bid,zid,sid,fname,_MLL_.DataType[dtype],
                                 tuple(imin),tuple(imax))  
    # --------------------------------------------------
    def fieldwrite(self,bid,zid,sid,dtype,fname,darray):
      """-Create a data array for a solution field -Flow Solution

         'field-id:I='*fieldwrite*'(base-id:I,zone-id:I,sol-id:I,args...)'

         The trailing args are 'data-type:S', 'field-name:S' and the
         array of data 'data-array:A'. See also 'coordwrite' remarks.
      """
      if dtype not in _MLL_.DataType.keys():
        raise ERR.CGNS_BadDataType
      return self.__db.fieldWrite(bid,zid,sid,fname,_MLL_.DataType[dtype],darray)
    # --------------------------------------------------
    def fieldinfo(self,bid,zid,sid,fid):
      """-Get infos about a given solution field -Flow Solution

         '(data-type:S,field-name:S)='*fieldinfo*'(base-id:I,zone-id:I,sol-id:I,field-id:I)'

         See 'coordwrite' remarks.
      """
      return self.__db.fieldInfo(bid,zid,sid,fid)
    # --------------------------------------------------
    def fieldid(self,bid,zid,sid,fid):
      """-Get the ADF id of a solution field -undocumented MLL

         'field-id:D='*fieldid*'(base-id:I,zone-id:I,sol-id:I,field-id:I)'

         See 'baseid' remarks.
      """
      if self.__modestring not in [_MLL_.MODE_READ,_MLL_.MODE_MODIFY]:
          raise ERR.CGNS_FileShouldBeReadable
      return self.__db.fieldId(bid,zid,sid,fid)
    # --------------------------------------------------
    def convergencewrite(self,nit,name):
      """-Craete of update a convergence node -Convergence

         'None='*convergencewrite*'(number-of-iteration:I,node-name:S)'

         Uses the current node. Should return the node id.
      """
      return self.__db.convergenceWrite(nit,name)
    # --------------------------------------------------
    def convergenceread(self): # could be an attribute ?
      """-Get the convergence under current node -Convergence

         '(number-of-iteration:I,node-name:S)='*convergenceread*'()'

         No args, current node is used.
      """
      return self.__db.convergenceRead()
    # --------------------------------------------------
    def nholes(self,bid,zid):
      """-Get the count of overset holes -Overset Holes

         'number-of-holes:I='*nholes*'(base-id:I,zone-id:I)'

         Returns the number of overset holes in the current zone
      """
      return self.__db.nHoles(bid,zid)
    # --------------------------------------------------
    def holeinfo(self,bid,zid,hid):
      """-Get info from a given overset hole node -Overset Holes

         'return-tuple='*holeinfo*'(base-id:I,zone-id:I,hole-id:I)'

         Returns a tuple containing 'name:S' of the overset hole,
         'grid-location:I' of the returned set(s) of points,
         the 'point-set-type:I', the 'number-of-point-sets:I' and the
         /number-of-points-per-point-set:I'.
         Should be called before *holeread* in order to have array dimensions
         before allocation.
      """
      return self.__db.holeInfo(bid,zid,hid)
    # --------------------------------------------------
    def holewrite(self,bid,zid,hname,gloc,psett,ar):
     """-Create a new overset hole node -Overset Holes

         'hole-id:I='*holewrite*'(base-id:I,zone-id:I,hole-name:S,g-location:S,point-array:A)'

         The grid location is a string. The corresponding enumerate can be
         found using the cross dictionnary.
     """
     if gloc not in _MLL_.GridLocation.keys():
       raise ERR.CGNS_BadGridLocation
     if psett not in _MLL_.PointSetType.keys():
       raise ERR.CGNS_BadPointSetType
     return self.__db.holeWrite(bid,zid,hname,_MLL_.GridLocation[gloc],
                                _MLL_.PointSetType[psett],ar)
    # --------------------------------------------------
    def holeread(self,bid,zid,hid):
      """-Get info from a given overset hole node -Overset Holes

         'point-array:A'='*holeread*'(base-id:I,zone-id:I,hole-id:I)'

         Gets the array containing the points. Dimensions depens
         on the PointSetType (see *holeinfo*).
      """
      return self.__db.holeRead(bid,zid,hid)
    # --------------------------------------------------
    def nbc(self,bid,zid):
      """-Get the count of BC -Boundary Condition

         'number-of-bc:I='*nbc*'(base-id:I,zone-id:I)'

         No comment
      """
      return self.__db.nBoco(bid,zid)
    # --------------------------------------------------
    def bcinfo(self,bid,zid,bcid):
      """-Get info from a given BC -Boundary Condition

         'return-tuple='*bcinfo*'(base-id:I,zone-id:I,bc-id:I)'

         The result tuple has the following members, in that order.
         The 'name:S' of the node, its 'bc-type:I' and its 'point-set-type:I'.
         The 'number-of-points:I', the 'normal-index:(I,I,I)',
         the 'data-type:I' for the normals, the 'normal-flag:I' and the
         'number-of-bc-data-set:I'.
      """
      return self.__db.bocoInfo(bid,zid,bcid)
    # --------------------------------------------------
    def bcread(self,bid,zid,bcid):
      """-Read point and normal lists from a given BC -Boundary Condition

         '(point-list:A,normal-list:A)='*bcread*'(base-id:I,zone-id:I,bc-id:I)'

         Comment
      """
      return self.__db.bocoRead(bid,zid,bcid)
    # --------------------------------------------------
    def bcwrite(self,bid,zid,bcname,bctype,pttype,ptlist):
      """-Create a new BC -Boundary Condition

         'bc-id:I='*bcwrite*'(base-id:I,zone-id:I,args...)'

         The trailing arguments are the following, in that order.
         The 'bc-name:S', 'bc-type:I', 'bc-point-set-type:I', the
         'point-set-list:((I,I,I),...)'.
         The number of points in the point set list is deduced from
         the length of the point list, except if the point set type is
         PointRange. In that case, the number of points is forced to 2.
      """
      if bctype not in _MLL_.BCType.keys():
        raise ERR.CGNS_BadBCType
      if pttype not in _MLL_.PointSetType.keys():
        raise ERR.CGNS_BadPointSetType
      # split ptlist from array to list of tuples...
      tpl=[]
      for tp in ptlist:
        tpl.append(tuple(tp))
      return self.__db.bocoWrite(bid,zid,bcname,_MLL_.BCType[bctype],
                                 _MLL_.PointSetType[pttype],tpl)
    # --------------------------------------------------
    def bcid(self,bid,zid,bcid):
      """-Get the BC ADF id -undocumented MLL

         'bc-id:D='*bcid*'(base-id:I,zone-id:I,bc-id:I)'

         Used for ADF functions
      """
      return self.__db.bocoId(bid,zid,bcid)
    # --------------------------------------------------
    def bcnormalwrite(self,bid,zid,bcid,nindex,nflags,dt,nlist):
      """-Write the normals of a given BC -Boundary Condition

         'None='*bcnormalwrite*'(base-id:I,zone-id:I,bc-id:I,args...)'

         The trailing args are the following, in this order.
         The 'normal-index:(I,I,I)', 'normal-flag:I', 'data-type:I',
         'normal-list:((D,D,D),...).
         Caution: normal-flag is forced to FALSE, normal-list is not taken
         into account.
      """
      if dt not in _MLL_.DataType.keys(): raise ERR.CGNS_BadDataType
      return self.__db.bocoNormalWrite(bid,zid,bcid,
                                       nindex,nflags,_MLL_.DataType[dt],nlist)
    # --------------------------------------------------
    def bcdatasetwrite(self,bid,zid,bcid,dsname,dstype):
      """-Write the dataset set of a given BC  -Boundary Condition

         'dset-id:I='*bcdatasetwrite*'(base-id:I,zone-id:I,bc-id:I,dset-name:S,dset-type:I)'

         dset-type:BCType
      """
      if dstype not in _MLL_.BCType.keys(): raise ERR.CGNS_BadBCType
      return self.__db.bocoDatasetWrite(bid,zid,bcid,dsname,_MLL_.BCType[dstype])
    # --------------------------------------------------
    def bcdatasetread(self,bid,zid,bcid,dsid):
      """-Read the dataset set of a given BC -Boundary Condition

         'return-tuple='*bcdatasetread*'(base-id:I,zone-id:I,bc-id:I,dset-id:I)'

         The return tuple is the following.
         (dset-name:S,bc-data-type:I,dir-flag:I,neu-flag:I)
      """
      return self.__db.bocoDatasetRead(bid,zid,bcid,dsid)
    # --------------------------------------------------
    def bcdatawrite(self,bid,zid,bcid,dsid,bctype):
      """-Write the data in a BC dataset -Boundary Condition

         'None='*bcdatawrite*'(base-id:I,zone-id:I,bc-id:I,dset-id:I,bc-data-type:I)'

         bc-data-type:BCDataType
      """
      if bctype not in _MLL_.BCDataType.keys(): raise ERR.CGNS_BadBCDataType
      return self.__db.bocoDataWrite(bid,zid,bcid,dsid,_MLL_.BCDataType[bctype])
    # --------------------------------------------------
    def ndiscrete(self,bid,zid):
      """-Get count of discrete node -Discrete Data

         'number-of-discrete:I='*ndiscrete*'(base-id:I,zone-id:I)'

         No Comment
      """
      return self.__db.nDiscrete(bid,zid)
    # --------------------------------------------------
    def discretewrite(self,bid,zid,name):
      """-Create a new discrete data node -Discrete Data

         'dics-id:I='*discretewrite*'(base-id:I,zone-id:I,disc-name:S)'

         No Comment
      """
      return self.__db.discreteWrite(bid,zid,name)
    # --------------------------------------------------
    def discreteread(self,bid,zid,did):
      """-Get the name of discrete node -Discrete Data

         'disc-name:S='*discreteread*'(base-id:I,zone-id:I,disc-id:I)'

         No Comment
      """
      return self.__db.discreteRead(bid,zid,did)
    # --------------------------------------------------
    def ngrids(self,bid,zid):
      """-Count the number of grids -Grid

         'number-of-grids:I='*ngrids*'(base-id:I,zone-id:I)'

         Comment
      """
      return self.__db.ngrids(bid,zid)
    # --------------------------------------------------
    def gridwrite(self,bid,zid,name):
      """-Create a new grid node -Grid

         'grid-id:I='*function*'(base-id:I,zone-id:I,grid-name:S)'

         Comment
      """
      return self.__db.gridWrite(bid,zid,name)
    # --------------------------------------------------
    def gridread(self,bid,zid,did):
      """-Get the grid name -Grid

         'grid-name:S='*gridread*'(base-id:I,zone-id:I,grid-id:I)'

         Comment
      """
      return self.__db.gridRead(bid,zid,did)
    # --------------------------------------------------
    def nintegrals(self):
      """-Get count of integral data nodes -Integral Data

         'number-of-integral='*nintegrals*'()'

         Counts under current node
      """
      return self.__db.nIntegrals()
    # --------------------------------------------------
    def integralwrite(self,name):
      """-Create a new integral node -Integral Data

         'integral-id:I='*integralwrite*'(integral-name:S)'

         Under current node.
      """
      return self.__db.integralWrite(name)
    # --------------------------------------------------
    def integralread(self,id):
      """-Get the name of integral node -Integral Data

         'integral-name:S='*integralread*'(integral-id:I)'

         Under current node.
      """
      return self.__db.integralRead(id)
    # --------------------------------------------------
    def nuserdata(self):
      """-Count number of user data -User Data

         'number-of-userdata:I='*nuserdata*'()'

         Under current node.
      """
      return self.__db.nUserdata()
    # --------------------------------------------------
    def userdatawrite(self,name):
      """-Create a new userdata node -User Data

         'userdata-id:I='*userdatawrite*'(userdata-name:S)'

         Under current node, which should be set by a goto call.
         Actually returns the new UserData node id. It is supposed
         that the goto call is made by pyCGNS.
      """
      self.__db.userdataWrite(name)
      n=self.nuserdata()
#      i=-1
#      for nu in range(n):
#        p=self.__lastPath.append((_MLL_.UserDefinedData_t,nu+1))
#        x=self.userdataread(nu+1)
#        if (x[1] == name): i=nu+1
#      if (i == -1):
#        print name
#        raise ERR.CGNS_NoSuchUserData
      return n
    # --------------------------------------------------
    def userdataread(self,id):
      """-Get name of the given user data id -User Data

         '(userdata-id:i,userdata-name:S)='*userdataread*'(userdata-id:I)'

         Under current node.
      """
      return self.__db.userdataRead(id)
    # --------------------------------------------------
    def unitswrite(self,mass,leng,time,temp,angl):
      """-Create or update a units set under current node -Units and Dimensionals

         'None='*unitswrite*'(mass-u:S,length-u:S,time-u:S,temp-u:S,angle-u:S)'

         See remarks about the constants dictionnary, one can either use the
         defined strings, variables or their enumerates.
         *should be much more documented/checked here*
      """
      if mass not in _MLL_.MassUnits.keys():       raise ERR.CGNS_BadMassUnit
      if leng not in _MLL_.LengthUnits.keys():     raise ERR.CGNS_BadLengthUnit
      if time not in _MLL_.TimeUnits.keys():       raise ERR.CGNS_BadTimeUnit
      if temp not in _MLL_.TemperatureUnits.keys():raise ERR.CGNS_BadTemperatureUnit 
      if angl not in _MLL_.AngleUnits.keys():      raise ERR.CGNS_BadAngleUnit      
      return self.__db.unitsWrite(_MLL_.MassUnits[mass],
                                  _MLL_.LengthUnits[leng],
                                  _MLL_.TimeUnits[time],
                                  _MLL_.TemperatureUnits[temp],
                                  _MLL_.AngleUnits[angl])
    # --------------------------------------------------
    def unitsread(self):
      """-Get the units under current node -Units and Dimensionals

         '(mass-u:S,length-u:S,time-u:S,temp-u:S,angle-u:S)='*unitsread*'()'

         See 'unitswrite' remarks.
      """
      return self.__db.unitsRead()
    # --------------------------------------------------
    def dataclasswrite(self,dt):
      """-Create or update the dataclass under current node -Units and Dimensionals

         'None='*dataclasswrite*'(data-class:I)'

         The 'data-class' is a 'DataClass' enumerate.
      """
      if dt not in _MLL_.DataClass.keys():
        raise ERR.CGNS_BadDataClass
      return self.__db.dataclassWrite(_MLL_.DataClass[dt])
    # --------------------------------------------------
    def dataclassread(self):
      """-Get the dataclass under current node -Units and Dimensionals

         'data-class:I='*dataclassread*'()'

         See 'dataclasswrite' remarks.
      """
      return self.__db.dataclassRead()
    # --------------------------------------------------
    def exponentswrite(self,dt,v):
      """-Create or update the exponents -Units and Dimensionals

         'None='*exponentswrite*'(data-type:I,(D,D,D,D,D))'

         Exponents values are: Mass, Length, Time, Temperature, Angle.
      """
      if dt not in _MLL_.DataType.keys():
        raise ERR.CGNS_BadDataType
      return self.__db.exponentsWrite(_MLL_.DataType[dt],v)
    # --------------------------------------------------
    def exponentsread(self):
      """-Get the exponents values -Units and Dimensionals

         '(D,D,D,D,D)='*exponentsread*'()'

         See 'exponentswrite' remarks.
      """
      return self.__db.exponentsRead()
    # --------------------------------------------------
    def exponentsinfo(self):
      """-Get the exponents datatype -Units and Dimensionals

         'datatype:I='*exponentsinfo*'()'

         Python only handles double. Beware at write time, you
         can have double/single.
      """
      return self.__db.exponentsInfo()
     # --------------------------------------------------
    def conversionwrite(self,dt,v):
      """-Create or update the conversion factors -Units and Dimensionals

         'None='*conversionwrite*'(data-type:I,(D,D))'

         Conversion values are: scale, offset
      """
      if dt not in _MLL_.DataType.keys():
        raise ERR.CGNS_BadDataType
      return self.__db.conversionWrite(_MLL_.DataType[dt],v)
    # --------------------------------------------------
    def conversionread(self):
      """-Get the conversion values -Units and Dimensionals

         '(D,D)='*conversionread*'()'

         See 'conversionwrite' remarks.
      """
      return self.__db.conversionRead()
    # --------------------------------------------------
    def conversioninfo(self):
      """-Get the conversion datatype -Units and Dimensionals

         'datatype:I='*conversioninfo*'()'

         Python only handles double. Beware at write time, you
         can have double/single.
      """
      return self.__db.conversionInfo()
    # --------------------------------------------------
    def ordinalwrite(self,o):
      """-Create or update an ordinal node under current node -Ordinal

         'None='*ordinalwrite*'(ordinal:I)'

         Comment
      """
      return self.__db.ordinalWrite(o)
    # --------------------------------------------------
    def ordinalread(self):
      """-Get the ordinal under current node -Ordinal

         'ordinal:I='*ordinalread*'()'

         Comment
      """
      return self.__db.ordinalRead()
    # --------------------------------------------------
    def bcwallfunctionread(self,bid,zid,bcid):
      """-Get the type of BC wallfunction -Boundary Condition

         'WallFunctionType:I='*bcwallfunctionread*'(base-id:I,zone-id:I,bc-id:I)'

         Comment
      """
      return self.__db.bcWallFunctionRead(bid,zid,bcid)
    # --------------------------------------------------
    def bcwallfunctionwrite(self,bid,zid,bcid,type):
      """-Set the type of BC wallfunction -Boundary Condition

         'None='*bcwallfunctionwrite*'(base-id:I,zone-id:I,bc-id:I,type:I)'

         Type is WallFunctionType
      """
      if type not in _MLL_.WallFunctionType.keys():
        raise ERR.CGNS_BadWallFunctionType
      return self.__db.bcWallFunctionWrite(bid,zid,bcid,_MLL_.WallFunctionType[type])
    # --------------------------------------------------
    def bcarearead(self,bid,zid,bcid):
      """-Get the area parameters -Boundary Condition

         '(region:S,type:I,surf:D)='*bcarearead*'(base-id:I,zone-id:I,bc-id:I)'

         Type is AreaType
      """
      return self.__db.bcAreaRead(bid,zid,bcid)
    # --------------------------------------------------
    def bcareawrite(self,bid,zid,bcid,type,surf,region):
      """-Set the area parameters -Boundary Condition

         'None='*bcareawrite*'(base-id:I,zone-id:I,bc-id:I,type:I,surf:D,region:S)'

         Type is AreaType
      """
      if type not in _MLL_.AreaType.keys():
        raise ERR.CGNS_BadAreaType
      return self.__db.bcAreaWrite(bid,zid,bcid,_MLL_.AreaType[type],surf,region)
    # --------------------------------------------------
    def rindwrite(self,rind):
      """-Create of update the rind indices under current node -Grid

         'None='*rindwrite*'((imin:I,imax:I,jmin:I,jmax:I,kmin,kmax:I))'

         Uses the current node. Tuple depends on dimensions, J or K could
         be unused in the case of 1D, 2D. Always give 6 integers, set
         them to zero if you are not 3D
      """
      return self.__db.rindWrite(rind)
    # --------------------------------------------------
    def rindread(self):
      """-Get the rind indices under current node -Grid

         '(imin:I,imax:I,jmin:I,jmax:I,kmin,kmax:I)='*rindread*'()'

         See 'rindwrite' comment.
      """
      return self.__db.rindRead()
    # --------------------------------------------------
    def nconns(self,bid,zid):
      """-Get number of generalized connectivities -Grid Connectivity

         'number:I='*nconns*'(base-id:I,zone-id:I)'

         No comment.
      """
      return self.__db.nConns(bid,zid)
    # --------------------------------------------------
    def conninfo(self,bid,zid,cid):
      """-Get information about generalized connect node -Grid Connectivity

         'return-tuple='*conninfo*'(base-id:I,zone-id:I,connect-id:I)'

         The return tuple contains: 'connect-name:S', 'gridlocation:I',
         'gridconnectivity:I', 'pointsettype:I', 'number-of-points:I',
         'donor-name:S', 'donor-zone-type:I', donor-point-set-type:I',
         'donor-data-type:I' and 'donor-number-of-points:I'
      """
      return self.__db.ConnInfo(bid,zid,cid)
    # --------------------------------------------------
    def connread(self,bid,zid,cid):
      """-Get generalized connectivity points -Grid Connectivity

         'return-tuple='*connread*'(base-id:I,zone-id:I,connect-id:I)'

         The tuple contains two arrays of integers 'target-interface-points:A'
         and 'donor-interface-oints:A'.
      """
      return self.__db.ConnRead(bid,zid,cid)
    # --------------------------------------------------
    def connwrite(self,bid,zid,name,gl,gt,pst,npt,pt,dname,dzt,dpst,ddt,dnpt,dpt):
      """-Create a generalized connectivity node -Grid Connectivity

         'connect-id:I='*connwrite*'(args)'

         The arguments are defining (in this order) the
         'base-id:I','zone-id:I' of the new node, its 'name:S', the
         'gridlocation:I', 'gridconnectivitytype:I' and 'point-set-type:I'
         of the current (target) node interface. The 'number-of-points:I' and
         'interface-points:A' which is an array of integers.
         Then the 'donor-name:S', its 'zonetype:I', 'point-set-type:I' and
         'data-type:I', the 'number-of-donor-points:I' and the actual array
         of points 'donor-points:A'.

         Note the 'DataType' is force to 'Integer'.
      """
      if gl not in _MLL_.GridLocation.keys():
        raise ERR.CGNS_BadGridLocation
      if ddt not in _MLL_.DataType.keys():
        raise ERR.CGNS_BadDataType
      if pst not in _MLL_.PointSetType.keys():
        raise ERR.CGNS_BadPointSetType
      if dpst not in [_MLL_.PointListDonor,
                      _MLL_.PointRangeDonor,
                      _MLL_.CellListDonor]:
        raise ERR.CGNS_BadPointSetType
      if gt not in _MLL_.GridConnectivityType.keys():
        raise ERR.CGNS_GridConnectivityType
      if dzt not in _MLL_.ZoneType.keys():
        raise ERR.CGNS_ZoneType
      return self.__db.ConnWrite(bid,zid,name,
                                 _MLL_.GridLocation[gl],
                                 _MLL_.GridConnectivityType[gt],
                                 _MLL_.PointSetType[pst],npt,pt,dname,
                                 _MLL_.ZoneType[dzt],
                                 _MLL_.PointSetType[dpst],
                                 _MLL_.DataType[ddt],dnpt,dpt)
    # --------------------------------------------------
    def connaverageread(self,bid,zid,cid):
      """-Get special connect properties -Special Grid Connectivity

         'averate-type:I='*connaverageread*'(base-id:I,zone-id:I,connect-id:I)'

         The type is 'AverageInterfaceType'.
      """
      return self.__db.ConnAverageRead(bid,zid,cid)
    # --------------------------------------------------
    def connaveragewrite(self,bid,zid,cid,at):
      """-Set special connect properties -Special Grid Connectivity

         'None='*connaveragewrite*'(base-id:I,zone-id:I,connect-id:I,averate-type:I)'

         The type is 'AverageInterfaceType'.
      """
      if at not in _MLL_.AverageInterfaceType.keys():
        raise ERR.CGNS_BadAverageInterfaceType
      return self.__db.ConnAverageWrite(bid,zid,cid,
                                        _MLL_.AverageInterfaceType[at])
    # --------------------------------------------------
    def connperiodicread(self,bid,zid,cid):
      """-Get special connect properties -Special Grid Connectivity

         'return-tuple='*connperiodicread*'(base-id:I,zone-id:I,connect-id:I)'

         The size of arrays is the base physical dimension. The reurn-tuple is
         '(rot-center:A,rot-angle:A,translation:A)'.
      """
      return self.__db.ConnPeriodicRead(bid,zid,cid)
    # --------------------------------------------------
    def connperiodicwrite(self,bid,zid,cid,rc,ra,tt):
      """-Set special connect properties -Special Grid Connectivity

         'None='*connperiodicwrite*'(base-id:I,zone-id:I,connect-id:I,args...)'

         the trailing args are 'rot-center:A', 'rot-angle:A' , 'translation:A'.
         The size of these arrays is the base physical dimension.
      """
      return self.__db.ConnPeriodicWrite(bid,zid,cid,rc,ra,tt)
    # --------------------------------------------------
    def none2oneglobal(self,bid):
      """-Count the one2one nodes for the whole base -Grid Connectivity

         'number-of-one2one:I='*none2oneglobal*'(base-id:I)'

         Comment
      """
      return self.__db.nOneToOneGlobal(bid)
    # --------------------------------------------------
    def none2one(self,bid,zid):
      """-Count the one2one nodes -Grid Connectivity

         'number-of-one2one:I='*none2one*'(base-id:I,zone-id:I)'

         Comment
      """
      return self.__db.nOneToOne(bid,zid)
    # --------------------------------------------------
    def one2oneid(self,bid,zid,id):
      """-Get the ADF id of a one2one node -undocumented MLL

         'one2one-id:D='*one2oneid*'(base-id:I,zone-id:I,one2one-id:I)'

         Comment
      """
      return self.__db.oneToOneId(bid,zid,id)
    # --------------------------------------------------
    def one2oneread(self,bid,zid,id):
      """-Get the one2one node informations -Grid Connectivity

         'return-tuple='*one2oneread*'(base-id:I,zone-id:I,one2one-id:I)'

         The return tuple has the following members, in that order. The
         'name:S' of the node, the 'donor-name:S', the 'range' tuple and
         the 'donor-range' tuple which are both six-integer tuples. Then
         the 'transform' tuple is a three-integer tuple.
      """
      return self.__db.oneToOneRead(bid,zid,id)
    # --------------------------------------------------
    def one2onereadglobal(self,bid):
      """-Get the one2one node informations for whole base -Grid Connectivity

         'return-list='*one2onereadglobal*'(base-id:I)'

         The return is a list of tuples, each tuple 
         tuple has the following members, in that order. The
         'name:S' of the node, the 'zone:S' name for which the conectivity
         information is related, the 'donor-name:S', the 'range' tuple and
         the 'donor-range' tuple which are both six-integer tuples. Then
         the 'transform' tuple is a three-integer tuple.
      """
      return self.__db.OneToOneReadGlobal(bid)
    # --------------------------------------------------
    def one2onewrite(self,bid,zid,name,donor,wrange,wdonorrange,transform):
      """-Create a 1to1 connectivity node -Grid Connectivity

         'one2one-id:I='*one2onewrite*'(base-id:I,zone-id:I,args...)'

         The trailing arguments are the following, in that order. The
         'name:S' of the node, the 'donor-name:S', the 'range' tuple and
         the 'donor-range' tuple which are both six-integer tuples. Then
         the 'transform' tuple is a three-integer tuple.
      """
      return self.__db.oneToOneWrite(bid,zid,name,donor,
                                     wrange,wdonorrange,transform)
    # --------------------------------------------------
    def nfamilies(self,bid): # could be an attribute ?
      """-Count families in the base -Families

         'number-of-families:I='*nfamilies*'(base-id:I)'

         No Comment
      """
      return self.__db.nFamilies(bid)
    # --------------------------------------------------
    def familyread(self,bid,fid):
      """-Get info about a given family -Families

         '(fam-name:S,number-of-fam-bc:I,number-of-geo:I)='*familyread*'(base-id:I,fam-id:I)'

         No Comment
      """
      return self.__db.familyRead(bid,fid)
    # --------------------------------------------------
    def familywrite(self,bid,name):
      """-Create a new family node -Families

         'fam-id:I='*familywrite*'(base-id:I,fam-name:S)'

         No Comment
      """
      return self.__db.familyWrite(bid,name)
    # --------------------------------------------------
    def familynameread(self):
      """-Get name of current node family -Families

         'fam-name:S='*familynameread*'()'

         No Comment
      """
      return self.__db.familyNameRead()
    # --------------------------------------------------
    def familynamewrite(self,name):
      """-Creates name of current node family -Families

         'None='*familynamewrite*'(fam-name:S)'

         No Comment
      """
      return self.__db.familyNameWrite(name)
    # --------------------------------------------------
    def familybocoread(self,bid,fid,bcid):
      """-Get name of current node family for a given BC -Families

         '(bc-name:S,bc-type:I)='*familybocoread*'(base-id:I,fam-id:I,bc-id:I)'

         No Comment
      """
      return self.__db.familyBocoRead(bid,fid,bcid)
    # --------------------------------------------------
    def familybocowrite(self,bid,fid,name,btype):
      """-Create name of current node family for a given BC -Families

         'bc-id:I='*familyboconamewrite*'(base-id:I,fam-id:I,bc-name:S,bc-type:I)'

         No Comment
      """
      if btype not in _MLL_.BCType.keys(): raise ERR.CGNS_BadBCType
      return self.__db.familyBocoWrite(bid,fid,name,_MLL_.BCType[btype])
    # --------------------------------------------------
    def georead(self,bid,fid,gid):
      """-Get geometry info -Families

         '(geo-name:S,file:S,CAD:S,parts:I)='*georead*'(base-id:I,fam-id:I,geo-id:I)'

         No Comment
      """
      return self.__db.geoRead(bid,fid,gid)
    # --------------------------------------------------
    def geowrite(self,bid,fid,gname,fname,cadname):
      """-Creates geometry info -Families

         'geo-id:I='*geowrite*'(base-id:I,fam-id:I,geo-name:S,file:S,CAD:S)'

         No Comment
      """
      return self.__db.geoWrite(bid,fid,gname,fname,cadname)
    # --------------------------------------------------
    def partread(self,bid,fid,gid,pid):
      """-Get part info -Families

         'part-name:S='*partread*'(base-id:I,fam-id:I,geo-id:I,part-id:I)'

         No Comment
      """
      return self.__db.partRead(bid,fid,gid,pid)
    # --------------------------------------------------
    def partwrite(self,bid,fid,gid,pname):
      """-Create part info -Families

         'part-id:I='*partwrite*'(base-id:I,fam-id:I,geo-id:I,part-name:S)'

         No Comment
      """
      return self.__db.partWrite(bid,fid,gid,pname)
    # --------------------------------------------------
    def biterread(self,bid):
      """-Get the name of the iterative data base -Iterative Data

         '(name:S,number-of-it:I)='*biterread*'(base-id:I)'

         No Comment
      """
      return self.__db.biterRead(bid)
    # --------------------------------------------------
    def biterwrite(self,bid,name,it):
      """-Create base iterative data  -Iterative Data

         'None='*biterwrite*'(base-id:I,name:S,iteration:I)'

         No Comment
      """
      return self.__db.biterWrite(bid,name,it)
    # --------------------------------------------------
    def ziterread(self,bid,zid):
      """-Get the name of the iterative data zone -Iterative Data

         'name:S='*ziterread*'(base-id:I,zone-id:I)'

         No Comment
      """
      return self.__db.ziterRead(bid,zid)
    # --------------------------------------------------
    def ziterwrite(self,bid,zid,name):
      """-Create zone iterative data -Iterative Data

         'None='*ziterwrite*'(base-id:I,zone-id:I,name:S)'

         No Comment
      """
      return self.__db.ziterWrite(bid,zid,name)
    # --------------------------------------------------
    def narrays(self): # could be an attribute ?
      """-Count arrays under the current node -Data Array

         'number-of-arrays:I='*narrays*'()'

         Use 'goto' to set the current node.
      """
      return self.__db.nArrays()
    # --------------------------------------------------
    def arrayread(self,aid):
      """-Get the array data -Data Array

         'data-array:A='*arrayread*'(array-id:I)'

         The array id its index under the current node.
         See 'arrayinfo' remarks.
      """
      return self.__db.arrayRead(aid)  
    # --------------------------------------------------
    def arraywrite(self,name,dtype,dim,ddim,darray):
      """-Create or update a new array -Data Array

         'array-id:I='*arraywrite*'(array-name:S,d-type:S,d-dim:I,d-vector:(I,...),d-array:A)'

         All *type*, *dim* *vector* are refering to the *array* of data
         itself. See also 'arrayinfo' remarks.
      """
      if dtype not in _MLL_.DataType.keys(): raise ERR.CGNS_BadDataType
      return self.__db.arrayWrite(name,_MLL_.DataType[dtype],dim,ddim,darray)
    # --------------------------------------------------
    def arrayinfo(self,aid):
      """-Get infos about a given array -Data Array

         '(array-name:S,data-type:S,data-dim:I,data-vector:(I,...))='*arrayinfo*'(array-id:I)'

         The current node is the parent node of the requested array.
         The data-type enumerate can be found using the cross dictionnary.
         There is redondancy of 'data-dim' and 'data-vector', first can be
         deduced from the second.
      """
      return self.__db.arrayInfo(aid)
    # --------------------------------------------------
    def lasterror(self):
      """-Get the current error -(pyCGNS)

         '(error-code:I,error-message:S)='*lasterror*'()'

         The 'error' attribute has the same contents.
      """
      return self.__db.lastError()
    # --------------------------------------------------
    def bases(self):
      """-Count number of bases -Base

         'nbases:I='*bases*'()'

         The 'nbases' attribute has the same contents.
      """
      return self.__db.nbases
    # --------------------------------------------------
    def descriptors(self):
      """-Count number of descriptors -Descriptor

         'ndescriptor:I='*descriptors*'()'

         The 'ndescriptor' attribute has the same contents.
      """
      return self.__db.ndescriptors
    # --------------------------------------------------
    def libversion(self):
      """-Get library version -(pyCGNS)

         'version:D='*libversion*'()'

         The 'version' attribute has the same contents.
      """
      return self.__db.version
    # --------------------------------------------------
    def checkzonesize(self,bsize,zsize,ztype):
      szlist=[0,0,0,0,0,0,0,0,0]
      for n in range(len(zsize)):
        szlist[n]=zsize[n]
#       if (ztype == _MLL_.Structured):
#         if (bsize==3):
#           if (len(sizelist) == 3):
#             sizelist+=[sizelist[0]-1,sizelist[1]-1,sizelist[2]-1,0,0,0]
#           elif (len(sizelist) == 6):
#             sizelist+=[0,0,0]
#           elif (len(sizelist) == 9):
#             pass
#       if (ztype == _MLL_.Unstructured):
#         if (len(sizelist) != 3): raise ValueError
#         if ((bsize==1) and (len(sizelist) not in [3,2,1])): raise ValueError
      return szlist
    # --------------------------------------------------
    def name(self):
      """-Get file name -(pyCGNS)

         'filename:S='*name*'()'

         The 'name' attribute has the same contents.
      """
      return self.__name
    # --------------------------------------------------
    def adfroot(self):
      """-Get ADF root id -(pyCGNS)

         'root-id:D='*adfroot*'()'

         The 'root' attribute has the same contents.
      """
      return self.__name
    # --------------------------------------------------    
    def __repr__(self):
      return '<pyCGNS instance from "%s">'%self.__name
    # --------------------------------------------------
    def __getattr__(self,name):
      if (name == 'root'):          return self.__db.rootid
      if (name == 'version'):       return self.__db.version        
      if (name == 'nbases'):        return self.__db.nbases
      if (name == 'ndescriptors'):  return self.__db.ndescriptors
      if (name == 'name'):          return self.__name
      if (name == 'error'):         return self.lasterror()
      else:
        return self.__dict__[name]
    # --------------------------------------------------    
    def repr(self): # who cares ?
      return "{ 'name': %s, 'mode': %d }"%(self.__name,self.__modestring)

# --- last line

