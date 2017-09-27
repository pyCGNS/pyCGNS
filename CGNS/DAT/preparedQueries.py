#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
# pyDAX - DBMS schema - Manages connection (high level)
# ------------------------------------------------------------
#
from __future__ import print_function
from CGNS.DAT.utils import *
import string

#
# ------------------------------------------------------------
# Print the first header, then print attribute values
# Print the header is not the same as previous one
#
PQPBaseQuery = """\
select distinct b.name,b.title,b.nzones,b.simulation,
       p.fileversion,p.filerelease,p.filechange,
       s.modificationdate,
       l.path,a.path,a.a_value,
       b.remarks,b.description,
       p.checkout_ct, p.checkin_ct, p.export_ct,p.import_ct,
       p.update_ct, p.modified
from cgnsBaseInfo as b, cgnsFlowEquationSetInfo as f,
     cgnsAttributeList as l, cgnsAttribute as a,
     cgnsPDMData as p, cgnsSystem as s
where b.name      = "%s"
  and p.entry_id  = b.entry_id
  and s.entry_id  = b.entry_id
  and f.id        = b.floweq_id
  and l.id        = f.attlist_id
  and a.alist_id  = l.id
order by b.name,a.path
"""


def PQpBaseCompact(s):
    PQpBase(s, compact=1)


#
def PQpBase(s, compact=None):
    if (compact):
        hp = """\
%(id)s [v%(version)s.%(release)s.%(change)s - %(mdate)s]
%(title)s
%(zones)s zone(s)
"""
    else:
        hp = """\
%(line)s
 - %(id)s - v%(version)s.%(release)s.%(change)s - %(mdate)s 
 - %(title)s
%(line)s
 Description:
%(description)s
%(line)s
 zones - checkout checkin   export   import   update
%(zones)4s %(co)8s %(ci)8s  %(ex)8s %(im)8s %(up)8s
%(line)s
 Remarks:
%(remarks)s
%(line)s
 Attributes:
 
"""
    if (compact):
        ap = """\
%(attribute)32.32s = %(value)s
"""
    else:
        ap = """\
 - %(attribute)s = %(value)s
"""
    k2 = ""
    for e in s:
        d = {}
        d['line'] = "-" * 60
        d['id'] = e[0]
        d['title'] = e[1]
        d['zones'] = e[2]
        d['version'] = e[4]
        d['release'] = e[5]
        d['change'] = e[6]
        d['mdate'] = e[7]
        if compact:
            d['attribute'] = e[9].split('/')[-1]
        else:
            d['attribute'] = e[9]
        d['value'] = asReportString(transAsPossible(e[10]))
        d['remarks'] = e[11]
        d['description'] = e[12]
        d['co'] = e[13]
        d['ci'] = e[14]
        d['ex'] = e[15]
        d['im'] = e[16]
        d['up'] = e[17]
        if e[18] == 1:
            d['up'] = str(d['up']) + '*'
        #
        k1 = e[0]
        if k1 != k2:
            k2 = k1
            print(hp % d,)
        print(ap % d,)


#
def PQpLinkList(s):
    k1 = None
    for kl in s:
        if (kl[0] != k1):
            print("%s %s" % (kl[0], '=' * 50))
            k1 = kl[0]
        print(" %s/%s\n -> [%s]%s" % (kl[2], kl[3], kl[1], kl[4]))


data = {
    # ------------------------------------------------------------
    'entries': ["""\
select f.id,f.status,f.filedate,f.fileid
from cgnsEntry as f""",
                "List all entries found in the DAX database (cgnsEntry)",
                "%3s %-9s %s %s"],
    # ------------------------------------------------------------
    'bases': ["""\
select e.filehaslink,e.status,e.filesize,b.name,b.title
from cgnsEntry as e, cgnsBaseInfo as b
where e.id=b.id""",
              "List a summary of all CGNS Bases found in the DAX database",
              "%.1d %.1s %10s %s %s"],
    # ------------------------------------------------------------
    'base': [PQPBaseQuery,
             "Describes the selected CGNS Base",
             PQpBase],
    # ------------------------------------------------------------
    'b': [PQPBaseQuery,
          "Describes the selected CGNS Base in a compact form",
          PQpBaseCompact],
    # ------------------------------------------------------------
    'log': ["""\
select connection,stamp,log from cgnslog
order by id""",
            "Database log output",
            "%10s %s %s"],
    # ------------------------------------------------------------
    'stat': ["""\
select f.filechecksum, p.fileversion, p.filerelease, p.filechange,
p.checkin_ct, p.checkout_ct, p.update_ct, p.export_ct, p.import_ct,
f.fileid
from cgnsEntry as f, cgnsPDMData as p
where p.entry_id=f.id""",
             "Give statistics for a given base entry",
             "%s v%s.%s.%s%3d%3d%3d%3d%3d %s"],
    # ------------------------------------------------------------
    'links': ["""\
select f.fileid, l.linkfile, l.localpath, l.localnode, l.linknode
from cgnsEntry as f, cgnsLink as l
where f.filehaslink=1 and l.entry_id=f.id""",
              "List all links for entries",
              PQpLinkList],
    # ------------------------------------------------------------
}
