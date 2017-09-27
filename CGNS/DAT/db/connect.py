#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------

# ------------------------------------------------------------
# pyDAX - DBMS schema - Manages connection (high level)
# ------------------------------------------------------------
#
# DBMS connection and drivers
# - The DB2 API actually does the job...
#
from   CGNS.DAT.utils import *
from   CGNS.DAT.exceptions import *
import CGNS.DAT.db.dbdrivers    as dxDD
import CGNS.DAT.preparedQueries as dxPQ
import CGNS.DAT.db.parseCGNS    as dxPS
import CGNS.DAT.db.foreignKeys  as dxFK
import CGNS.DAT.db.db_owner     as ow
import numpy            as N
import string
import os
import tarfile
import tempfile

#
(dxModeCreateOnly, dxModeCreateCheckin, dxModeUpdateCheckin) = range(3)


# ------------------------------------------------------------
class daxDB:
    __logCount = 1
    """\
    daxDB class -
    """

    # ----------------------------------------------------------
    def __init__(self, baseconnection):
        if ((baseconnection)
            and (type(baseconnection) == type((0,)))
            and (len(baseconnection) == 3)
            and (type(baseconnection[0]) == type(""))
            and (type(baseconnection[1]) == type(""))
            and (type(baseconnection[2]) == type(""))):
            self.__base = baseconnection[0]
            self.__user = baseconnection[1]
            self.__pswd = baseconnection[2]
        else:
            raise DAXBadConnectionString(baseconnection)
        self.__log = []
        self.__connected = 0

    # ----------------------------------------------------------
    def __del__(self):
        if (self.__dict__.has_key('_db')):
            self._db.rollback()  # Abort transaction

    # ----------------------------------------------------------
    def _connect(self):
        if (self.__connected): return self  # a transaction is still running
        self._db = dxDD.daxDriverDefault(self.__base, self.__user, self.__pswd)
        stmt = "set autocommit = 0"
        self._db.runSQL(stmt)
        stmt = "start transaction"
        self._db.runSQL(stmt)
        self.__connected = 1
        self._addLog("Start transaction")
        return self

    # ----------------------------------------------------------
    def _deconnect(self):
        self._db.commit()
        self._addLog("Commit")
        del self._db
        self.__connected = 0
        # ----------------------------------------------------------

    def _addLog(self, msg):
        ptrace(msg, level=1)
        self.__log.append("#%.3d#%s" % (daxDB.__logCount, msg))
        tp = (self.__user, date(), msg)
        stmt = """insert into cgnsLog (connection,stamp,log) \
            values ("%s","%s","%s")""" % tp
        self._db.runSQL(stmt)
        daxDB.__logCount += 1
        return daxDB.__logCount

    # ----------------------------------------------------------
    def __reloadMeta(self, id):
        dct = {}
        stmt = """select id,status from cgnsEntry where fileid="%s" """ % id
        (eid, status) = self._db.runSQL(stmt)[0]
        stmt = """select fileversion,filerelease,filechange \
            from cgnsPDMData where entry_id='%s'""" % eid
        r = self._db.runSQL(stmt)[0]
        dct[dxPS.version] = int(r[0])
        dct[dxPS.release] = int(r[1])
        dct[dxPS.change] = int(r[2])
        dct[dxPS.status] = str(status)
        stmt = 'update cgnsPDMData set modified=0 where entry_id=%s' % (eid)
        self._db.runSQL(stmt)
        return dct

    # ----------------------------------------------------------
    def __updateBase(self, id, dct, mode):
        # --- get entry id
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        # --- create Base info entry
        if (mode in [dxModeCreateOnly, dxModeCreateCheckin]):
            stmt = """insert into cgnsBaseInfo(name,title,description,
      remarks,physicaldim,celldim,nzones,simulation,entry_id)
      values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        # --- update exiting Base info entry
        else:
            stmt = """update cgnsBaseInfo set name=%s ,title=%s, description=%s,
      remarks=%s, physicaldim=%s, celldim=%s, nzones=%s, simulation=%s
      where entry_id=%s"""
        tp = ((dct[dxPS.basename],
               dct[dxPS.basetitle],
               dct[dxPS.basedescription],
               dct[dxPS.baseremarks],
               dct[dxPS.pdim],
               dct[dxPS.cdim],
               dct[dxPS.nzones],
               dct[dxPS.simulation],
               eid),)
        self._db.runSQL(stmt, values=tp)

    # ----------------------------------------------------------
    def __updateMeta(self, id, dct, mode):
        # --- get entry id
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        # --- creation if first checkin
        if mode in [dxModeCreateOnly, dxModeCreateCheckin]:
            stmt = """insert into cgnsSystem(entry_id) values (%s)"""
            self._db.runSQL(stmt, values=((eid,),))
            stmt = """insert into cgnsPDMData(fileversion,filerelease,filechange,
              entry_id,modified) values (%s,%s,%s,%s,%s)"""
            self._db.runSQL(stmt, values=((transAsPossible(dct[dxPS.version]),
                                           transAsPossible(dct[dxPS.release]),
                                           transAsPossible(dct[dxPS.change]), eid, 0),))
        else:
            stmt = """update cgnsPDMData set fileversion="%s", \
                                     filerelease="%s", \
                                     filechange="%s"
              where entry_id=%s""" % \
                   (transAsPossible(dct[dxPS.version]),
                    transAsPossible(dct[dxPS.release]),
                    transAsPossible(dct[dxPS.change]), eid)
            self._db.runSQL(stmt)
        stmt = """select id from cgnsPDMData where entry_id='%s'""" % eid
        pid = self._db.runSQL(stmt)[0][0]
        stmt = """update cgnsSystem set pdm_id=%s where entry_id=%s""" % (pid, eid)
        self._db.runSQL(stmt)
        stmt = """update cgnsSystem set creationdate="%s" where entry_id=%s""" \
               % (dct[dxPS.creationdate], eid)
        self._db.runSQL(stmt)
        stmt = """update cgnsSystem set modificationdate="%s" where entry_id=%s""" \
               % (dct[dxPS.modificationdate], eid)
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def _updateLinks(self, id, dct, mode):
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        if (dct[dxPS.linkslist]):
            stmt = """update cgnsEntry set filehaslink=1 where id=%s""" % eid
            self._db.runSQL(stmt)
        # --- in case of update, it's easier to delete all links
        #     no other table should refer to this link table
        if (mode == dxModeUpdateCheckin):
            stmt = """delete from cgnsLink where entry_id=%s""" % eid
            self._db.runSQL(stmt)
        for lk in dct[dxPS.linkslist]:
            pth = lk[0].split('/')
            tp = (eid, str.join('/', pth[:-1]), pth[-1], lk[1], lk[2])
            stmt = """insert into cgnsLink (entry_id, \
                                    localpath,\
                                    localnode,\
                                    linkfile, \
                                    linknode)
              values (%s,%s,%s,%s,%s)"""
            self._db.runSQL(stmt, values=(tp,))
            # --- link to existing entries
            lid = xn(lk[1])  # cut trailing .cgns
            elid = self.getEntryId(lid)
            # TWO OPTIONS:
            # 1- we decide to avoid raising an exception when the linked-to id is
            # not present. This is easier for user, for example when there
            # is a loop in the links.
            # 2- we raise an exception, no uncomplete data is allowed
            #
            # today special offer: option 2
            #
            if (not elid):
                raise DAXNoSuchIdentifier(lid)
            stmt = """select id from cgnsLink where entry_id=%s \
                                        and localpath="%s" \
                                        and localnode="%s" """ % \
                   (eid, str.join('/', pth[:-1]), pth[-1])
            nlid = self._db.runSQL(stmt)[0][0]
            stmt = """update cgnsLink set linked_id='%s' where id=%s""" % (elid, nlid)
            self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def __updateReference(self, id, dct, mode):
        pass

    # ----------------------------------------------------------
    def __updateFlowEquationSet(self, id, dct, mode):
        # --- get entry id
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        # --- create Flowequation
        if (mode in [dxModeCreateOnly, dxModeCreateCheckin]):
            stmt = """insert into cgnsAttributeList(path,modified,entry_id)
                          values (%s,%s,%s)"""
            pth = "%s/%s" % (id, dxPS.flowequation)
            self._db.runSQL(stmt, values=((pth, 0, eid),))
            stmt = """select id from cgnsAttributeList where path='%s'""" % pth
            fid = self._db.runSQL(stmt)[0][0]
            stmt = """insert into cgnsFlowEquationSetInfo(base_id,attlist_id)
              values(%s,%s)"""
            self._db.runSQL(stmt, values=((eid, fid),))
            stmt = """select id from cgnsFlowEquationSetInfo where attlist_id=%s""" % fid
            qid = self._db.runSQL(stmt)[0][0]
            stmt = """update cgnsBaseInfo set floweq_id=%s where id=%s""" % (qid, eid)
            self._db.runSQL(stmt)
        # --- update FlowEquation
        else:
            stmt = """select f.attlist_id 
              from cgnsBaseInfo as b, cgnsFlowEquationSetInfo as f
              where b.entry_id=%s
                and b.floweq_id=f.id""" % eid
            fid = self._db.runSQL(stmt)[0][0]
            # ---
        for k in dct:
            # --- this table should not be empty, always add a watchdog
            if (k[:len(dxPS.flowequation)] == dxPS.flowequation):
                vv = transAsPossible(dct[k])
                if (mode in [dxModeCreateOnly, dxModeCreateCheckin]):
                    stmt = """insert into cgnsAttribute(alist_id,path,a_value)
                  values(%s,%s,%s)"""
                    self._db.runSQL(stmt, values=((fid, k, vv),))
                else:
                    stmt = """select id from cgnsAttribute
                  where alist_id=%s and path="%s" """ % (fid, k)
                    aidr = self._db.runSQL(stmt)
                    if (aidr):  # Found
                        aid = aidr[0][0]
                        stmt = 'update cgnsAttribute set a_value="%s" where id=%s ' % (vv, aid)
                        self._db.runSQL(stmt)
                    else:  # no such attribute in previous creation/update -> add it
                        stmt = """insert into cgnsAttribute(alist_id,path,a_value)
                    values(%s,%s,%s)"""
                        self._db.runSQL(stmt, values=((fid, k, vv),))
        if (mode in [dxModeCreateCheckin, dxModeUpdateCheckin]):
            stmt = 'select checkin_ct from cgnsPDMData where entry_id=%s' % (eid)
            ct = self._db.runSQL(stmt)[0][0] + 1
            stmt = 'update cgnsPDMData set checkin_ct="%s" where entry_id=%s' % (ct, eid)
            self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def __searchIDInDAX(self, id):
        r = 1
        stmt = """select fileid from cgnsEntry where fileid='%s'""" % id
        self._connect()
        r = self._db.runSQL(stmt)
        # if not r:
        #  self._addLog("Check ID is in DAX [%s] : FOUND"%(id))
        # else:
        #  self._addLog("Check ID is in DAX [%s] : NOT FOUND"%(id))
        # self._deconnect()
        return r

    # ----------------------------------------------------------
    def __changePDMVersion(self, eid, v, field='fileversion'):
        vv = transAsPossible(v)
        if (vv not in range(0, 100)):
            raise DAXIncorrectValues("Attribute [%s] Value [%s]" % (a, vv))
        tp = (field, vv, eid)
        stmt = """update cgnsPDMData set %s="%s" where entry_id=%s""" % tp
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def __changeBaseInfo(self, eid, v, field='title'):
        stmt = """update cgnsBaseInfo set %s="%s" where entry_id=%s""" % (field, v, eid)
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def __changeEntryStatus(self, eid, v):
        vv = transAsPossible(v)
        if (vv not in dxFK.FileStatusEnum):
            raise DAXIncorrectValues("Value [%s] is not a Status" % (vv))
        stt = self._getStatusP(eid)
        if (((stt != dxFK.UNKNOWN) and (v == dxFK.DRAFT))
            or ((stt != dxFK.DRAFT) and (v == dxFK.BASELINE))
            or ((stt != dxFK.BASELINE) and (v == dxFK.REFERENCE))
            or ((stt != dxFK.REFERENCE) and (v == dxFK.OBSOLETE))):
            raise DAXUpdateRefused("Cannot set status to [%s] on [%s] base" % (v, stt))
        tp = (vv, eid)
        stmt = """update cgnsEntry set status="%s" where id=%s""" % tp
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def getEntryId(self, id):
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        rst = self._db.runSQL(stmt)
        eid = None
        try:
            eid = rst[0][0]
        except IndexError:
            pass
        return eid

    # ----------------------------------------------------------
    def _getStatus(self, id):
        stmt = """select status from cgnsEntry where fileid="%s" """ % id
        st = self._db.runSQL(stmt)[0][0]
        return st

    # ----------------------------------------------------------
    def _getStatusP(self, eid):
        stmt = """select status from cgnsEntry where id="%s" """ % eid
        st = self._db.runSQL(stmt)[0][0]
        return st

    # ----------------------------------------------------------
    def _checkNoSuchIDInDAX(self, id):
        if (self.__searchIDInDAX(id)): raise DAXIdentifierAlreadyExist(id)

    # ----------------------------------------------------------
    def _checkIsIDInDAX(self, id):
        if (not self.__searchIDInDAX(id)): raise DAXNoSuchIdentifier(id)

    # ----------------------------------------------------------
    def _insertFileAndAttributes(self, filename, mode=dxModeCreateCheckin, nolk=0):
        # ---
        if (mode == dxModeUpdateCheckin):
            st = self._getStatus(xn(filename))
            if (st in [dxFK.REFERENCE, dxFK.OBSOLETE]):
                raise DAXCheckinRefused("Base has status [%s]" % st)
        if (mode in [dxModeCreateCheckin, dxModeUpdateCheckin]):
            # read and use actual file
            self._addLog("Load file and check %s" % (filename))
            cfile = dxPS.fileCGNS(filename)
            blobData = cfile.asBinaryString()
            ck = cfile.checksum()
            cfile.checkStructure()
            dct = cfile.parse()
            stt = getStat(filename)
            id = dct[dxPS.basename]
            dxPS.checkConsistency(dct)
        else:
            # use a dumb default dictionary
            id = xn(filename)
            cfile = dxPS.fileCGNS()
            blobData = cfile.asBinaryString()
            ck = cfile.checksum()
            dct = dxPS.parseCGNSfile(None)
            dct[dxPS.basename] = id
            stt = getStat(None)
        own = ow.getOwner(self._db, transAsPossible(dct[dxPS.owner]))
        if (mode in [dxModeCreateCheckin, dxModeCreateOnly]):
            # --- insert cgnsEntry data (creation only)
            self._addLog("Add new cgnsEntry for %s" % (id))
            tp = [id, len(blobData), ck, stt[-1], own, dxFK.PUBLIC, dxFK.DRAFT]
            stmt = """insert into cgnsEntry(fileid,      \
                                    filesize,    \
                                    filechecksum,\
                                    filedate,    \
                                    owner_id,    \
                                    policy,      \
                                    status)      \
            values(%s,%s,%s,%s,'%s',%s,%s)"""
            self._db.runSQL(stmt, values=(tuple(tp),))
        # ---
        oid = None
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        if (mode in [dxModeUpdateCheckin]):
            if ((st == dxFK.BASELINE) and (dct[dxPS.status] == dxFK.DRAFT)):
                raise DAXCheckinRefused("Base cannot go back to DRAFT status")
            if (dct[dxPS.status] not in [dxFK.DRAFT, dxFK.BASELINE]):
                raise DAXCheckinRefused("Cannot set [%s] status through checkin" % \
                                        dct[dxPS.status])
            tp = (len(blobData), ck, stt[-1], own, dxFK.PUBLIC, dxFK.DRAFT, eid)
            stmt = """update cgnsEntry set filesize="%s",    \
                                   filechecksum="%s",\
                                   filedate="%s",    \
                                   owner_id="%s",    \
                                   policy="%s",      \
                                   status="%s"       \
              where id=%s""" % tp
            self._db.runSQL(stmt)
            stmt = """select filedata_id from cgnsEntry where id="%s" """ % eid
            oid = self._db.runSQL(stmt)[0][0]
            stmt = """delete from cgnsBlobEntry where id=%s""" % (oid)
            self._db.runSQL(stmt)
        # --- create/update (dummy/actual) blob
        tp = [eid, blobData]
        stmt = """insert into cgnsBlobEntry(entry_id,filedata) values(%s,%s)"""
        self._db.runSQL(stmt, values=(tuple(tp),))
        stmt = """select id from cgnsBlobEntry where entry_id="%s" """ % eid
        bid = self._db.runSQL(stmt)[0][0]
        stmt = """update cgnsEntry set filedata_id=%s where id=%s""" % (bid, eid)
        self._db.runSQL(stmt)
        # --- update attributes over db schema
        self.__updateBase(id, dct, mode)
        self.__updateMeta(id, dct, mode)
        self.__updateReference(id, dct, mode)
        self.__updateFlowEquationSet(id, dct, mode)
        if (not nolk): self._updateLinks(id, dct, mode)
        return (id, dct, mode)
        # ----------------------------------------------------------

    def _changeFileInDirectory(self, id, path):
        # only meta info can be changed...
        dct = self.__reloadMeta(id)
        cfile = dxPS.fileCGNS()
        cfile.open(path, 1)  # write !
        cfile.update(dct)
        # ----------------------------------------------------------

    def _outputFileInDirectory(self, id, path):
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        stmt = """select filedata from cgnsBlobEntry where entry_id="%s" """ % eid
        ptrace("Reading blob [%s]" % id)
        blobData = self._db.runSQL(stmt)[0][0]
        ptrace("Opening file [%s]" % path)
        try:
            f = open(path, 'wb')
            f.write(blobData)
            f.close()
        except IOError:
            raise DAXDiskFull("Cannot checkout")
        stmt = 'select checkout_ct from cgnsPDMData where entry_id=%s' % (eid)
        ct = self._db.runSQL(stmt)[0][0] + 1
        stmt = 'update cgnsPDMData set checkout_ct="%s" where entry_id=%s' % (ct, eid)
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def _deleteFromDatabase(self, id):
        stmt = """select id from cgnsEntry where fileid="%s" """ % id
        eid = self._db.runSQL(stmt)[0][0]
        st = self._getStatusP(eid)
        if (st == dxFK.REFERENCE):
            raise DAXRemovalRefused("Base has status [REFERENCE]")
        # check links
        stmt = """select e.fileid from cgnsLink as l, cgnsEntry as e
            where l.linked_id=e.id and e.fileid='%s'""" % (id)
        rl = self._db.runSQL(stmt)
        if (rl): raise DAXRemovalRefused("Another base has a link to %s" % id)
        stmt = """delete from cgnsEntry where fileid="%s" """ % id
        self._db.runSQL(stmt)
        stmt = """delete from cgnsAttributeList where entry_id="%s" """ % eid
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def _loopOverUpdates(self, fileid, attlist):
        stmt = """select id from cgnsEntry where fileid="%s" """ % fileid
        eid = self._db.runSQL(stmt)[0][0]
        for r in range(len(attlist[0])):
            a, v = attlist[0][r], attlist[1][r]
            # --- look for attribute list
            stmt = """select id from cgnsAttributeList where entry_id=%s""" % eid
            aid = self._db.runSQL(stmt)[0][0]
            stmt = """update cgnsAttribute
              set a_value="%s"
              where alist_id=%s
              and path="%s" """ % (v, aid, a)
            self._db.runSQL(stmt)
            # --- specific PDM data (has an impact on DAX behaviour)
            if (a == dxPS.owner):
                pass
            elif (a == dxPS.version):
                self.__changePDMVersion(eid, v)
            elif (a == dxPS.release):
                self.__changePDMVersion(eid, v, field='filerelease')
            elif (a == dxPS.change):
                self.__changePDMVersion(eid, v, field='filechange')
            elif (a == dxPS.policy):
                pass
            elif (a == dxPS.status):
                self.__changeEntryStatus(eid, v)
            elif (a == dxPS.basetitle):
                self.__changeBaseInfo(eid, v)
            elif (a == dxPS.baseremarks):
                self.__changeBaseInfo(eid, v, field='remarks')
            elif (a == dxPS.basedescription):
                self.__changeBaseInfo(eid, v, field='description')
        stmt = 'select update_ct from cgnsPDMData where entry_id=%s' % (eid)
        ct = self._db.runSQL(stmt)[0][0] + 1
        stmt = 'update cgnsPDMData set modified=1 where entry_id=%s' % (eid)
        self._db.runSQL(stmt)
        stmt = 'update cgnsPDMData set update_ct="%s" where entry_id=%s' % (ct, eid)
        self._db.runSQL(stmt)

    # ----------------------------------------------------------
    def __checkin(self, fileid, filepath, mode):
        self._connect()
        self._addLog("Check-in for %s [into %s]" % (fileid, filepath))
        self._insertFileAndAttributes(dn(filepath, fileid), mode)
        self._deconnect()

    # ----------------------------------------------------------
    def __checkout(self, fileid, filepath):
        self._connect()
        self._addLog("Check-out for %s [from %s]" % (fileid, filepath))
        self._outputFileInDirectory(fileid, dn(filepath, fileid))
        self._changeFileInDirectory(fileid, dn(filepath, fileid))
        self._deconnect()

    # ----------------------------------------------------------
    def __remove(self, fileid):
        self._connect()
        self._addLog("Remove entry %s" % (fileid))
        self._deleteFromDatabase(fileid)
        self._deconnect()

    # ----------------------------------------------------------
    def __update(self, fileid, attlist):
        self._connect()
        self._addLog("Update entry %s" % (fileid))
        self._loopOverUpdates(fileid, attlist)
        self._deconnect()

    # ============================================================
    # Interface
    # ----------------------------------------------------------
    def log(self, purge=0):
        # should purge DB !
        for n in range(0, min(purge, len(self.__log))):
            self.__log.pop(0)
        return self.__log

    # ----------------------------------------------------------
    def create(self, fileid, pathid=None):
        checkDAXID(fileid)
        mode = dxModeCreateOnly
        if (pathid):
            checkFile(fileid, pathid)
            mode = dxModeCreateCheckin
        self._checkNoSuchIDInDAX(fileid)
        self.__checkin(fileid, pathid, mode)
        return self

    # ----------------------------------------------------------
    def checkout(self, fileid, pathid=None):
        checkDAXID(fileid)
        if (not pathid): pathid = '.'
        try:
            checkFile(fileid, pathid)
            raise DAXCGNSFileAlreadyExist(dn(pathid, fileid))
        except DAXNoSuchCGNSFile:
            pass
        self._checkIsIDInDAX(fileid)
        self.__checkout(fileid, pathid)
        return self

    # ----------------------------------------------------------
    def checkin(self, fileid, pathid=None):
        checkDAXID(fileid)
        if (not pathid): pathid = '.'
        checkFile(fileid, pathid)
        # check if it exists, should be a new version
        self._checkIsIDInDAX(fileid)
        self.__checkin(fileid, pathid, mode=dxModeUpdateCheckin)

    # ----------------------------------------------------------
    def remove(self, fileid):
        checkDAXID(fileid)
        self._checkIsIDInDAX(fileid)
        self.__remove(fileid)
        return self

    # ----------------------------------------------------------
    def update(self, fileid, alist):
        checkDAXID(fileid)
        self._checkIsIDInDAX(fileid)
        dxPS.checkAttributes(alist[0])
        self.__update(fileid, alist)
        return self

    # ----------------------------------------------------------
    def usage(self):
        return self.__doc__


# ------------------------------------------------------------
class daxQT(daxDB):
    __preparedQueries = dxPQ.data
    """\
  daxQT class
  """

    # ----------------------------------------------------------
    def __init__(self, baseconnection):
        daxDB.__init__(self, baseconnection)

    # ============================================================
    # Interface
    # ----------------------------------------------------------
    def query(self, sqlstmt):
        self._connect()
        r = self._db.runSQL(sqlstmt)
        self._deconnect()
        return r

    # ----------------------------------------------------------
    def pquery(self, queryid, valuelist):
        self._connect()
        try:
            stmt = self.__preparedQueries[queryid][0] % tuple(valuelist)
        except KeyError:
            raise DAXPreparedQueryNotFound(queryid)
        except TypeError:
            raise DAXQueryFailed(queryid,
                                 self.__preparedQueries[queryid][0],
                                 valuelist)
        r = self._db.runSQL(stmt)
        self._deconnect()
        return self.__preparedQueries[queryid][2], r

    # ----------------------------------------------------------
    def list(self):
        return self.__preparedQueries

    # ----------------------------------------------------------
    def usage(self):
        return self.__doc__


# ------------------------------------------------------------
class daxET(daxDB):
    """\
  daxET class
  """

    # ----------------------------------------------------------
    def __init__(self, baseconnection):
        daxDB.__init__(self, baseconnection)

    # ----------------------------------------------------------
    def __export(self, fileid, pathid, compress):
        self._addLog("Export entry [%s]" % fileid)
        eid = self.getEntryId(fileid)
        stmt = 'select filehaslink from cgnsEntry where id=%s' % eid
        haslink = self._db.runSQL(stmt)[0][0]
        # use given path (we suppose there are disk space here !)
        filepath = tempfile.mkdtemp('', '.dax.tmp.', pathid)
        if (compress):
            cmode = 'w:gz'
        else:
            cmode = 'w'
        ft = tarfile.open(tn(filepath, fileid, compress), cmode)
        ft.posix = True
        self._outputFileInDirectory(fileid, dn(filepath, fileid))
        ft.add(dn(filepath, fileid), fn(fileid))
        stmt = 'select export_ct from cgnsPDMData where entry_id=%s' % (eid)
        ct = self._db.runSQL(stmt)[0][0] + 1
        stmt = 'update cgnsPDMData set export_ct="%s" where entry_id=%s' % (ct, eid)
        self._db.runSQL(stmt)
        idlist = []
        if (haslink):
            stmt = 'select distinct linked_id from cgnsLink where entry_id=%s' % eid
            idlist = self._db.runSQL(stmt)
            for lid in idlist:
                stmt = 'select fileid from cgnsEntry where id=%s' % lid[0]
                ri = self._db.runSQL(stmt)[0][0]
                self._outputFileInDirectory(ri, dn(filepath, ri))
                ft.add(dn(filepath, ri), fn(ri))
                stmt = 'select export_ct from cgnsPDMData where entry_id=%s' % (lid[0])
                ct = self._db.runSQL(stmt)[0][0] + 1
                stmt = 'update cgnsPDMData set export_ct="%s" \
              where entry_id=%s' % (ct, lid[0])
                self._db.runSQL(stmt)
                if (not compress):
                    os.remove(dn(filepath, ri))
        ft.close()
        # in compress mode, no way to delete files after insert...
        if (compress and haslink):
            for lid in idlist:
                stmt = 'select fileid from cgnsEntry where id=%s' % lid
                ri = self._db.runSQL(stmt)[0][0]
                os.remove(dn(filepath, ri))
        os.rename(tn(filepath, fileid, compress), tn(pathid, fileid, compress))
        os.remove(dn(filepath, fileid))
        os.removedirs(filepath)

    # ----------------------------------------------------------
    def __import(self, fileid, pathid, compress):
        self._addLog("Import entry [%s]" % fileid)
        # use given path (we suppose there are disk space here !)
        filepath = tempfile.mkdtemp('', '.dax.tmp.', pathid)
        if (compress):
            cmode = 'r:gz'
        else:
            cmode = 'r'
        ft = tarfile.open(tn(pathid, fileid, compress), cmode)
        ft.posix = True
        idlist = []
        for fte in ft:
            ft.extract(fte, filepath)
            # get id, check if exists or not
            fid = xn(fte.name)
            stmt = """select fileid from cgnsEntry where fileid='%s'""" % fid
            r = self._db.runSQL(stmt)
            if (r):
                mode = dxModeUpdateCheckin
            else:
                mode = dxModeCreateCheckin
            (id, dct, md) = self._insertFileAndAttributes(dn(filepath, fid), mode, nolk=1)
            idlist.append([id, dct])
            stmt = """select id from cgnsEntry where fileid='%s'""" % fid
            eid = self._db.runSQL(stmt)[0][0]
            stmt = 'select import_ct from cgnsPDMData where entry_id=%s' % (eid)
            ct = self._db.runSQL(stmt)[0][0] + 1
            stmt = 'update cgnsPDMData set import_ct="%s" where entry_id=%s' % (ct, eid)
            self._db.runSQL(stmt)
            os.remove(dn(filepath, fid))
        # update links once all entries are there
        for idl in idlist:
            self._updateLinks(idl[0], idl[1], mode)
        ft.close()
        os.removedirs(filepath)

    # ============================================================
    # Interface
    # ----------------------------------------------------------
    def exportTree(self, fileid, pathid, compress=''):
        checkDAXID(fileid)
        self._checkIsIDInDAX(fileid)
        if (pathid):
            checkPath(pathid)
        else:
            pathid = '.'
        if (compress):
            cp = 'gz'
        else:
            cp = ''
        try:
            checkExport(pathid, fileid, cp)
            raise DAXCGNSFileAlreadyExist(tn(pathid, fileid, cp))
        except DAXNoSuchCGNSFile:
            pass
        self._connect()
        self.__export(fileid, pathid, cp)
        self._deconnect()
        return self

    # ----------------------------------------------------------
    def importTree(self, fileid, pathid):
        checkDAXID(fileid)
        if (pathid):
            checkPath(pathid)
        else:
            pathid = '.'
        cp = 'gz'
        try:
            checkExport(pathid, fileid, cp)
        except DAXNoSuchCGNSFile:
            cp = ''
            checkExport(pathid, fileid, cp)
        self._connect()
        self.__import(fileid, pathid, cp)
        self._deconnect()
        return self

    # ----------------------------------------------------------
    def usage(self):
        return self.__doc__

#
# last line
