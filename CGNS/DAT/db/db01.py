#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# This file can be configured for your own existing DBMS
# Some attribute names can be changed (foreign keys)
# see the foreignKeys.py file
#
import foreignKeys as mapping

#
# --- SYNTAX *IS* SQLITE3
#
# ------------------------------------------------------------
cgnsEntryTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsEntry (
  -- ---------------------------------------------------------
  id           INT UNSIGNED,
  owner_id     INT UNSIGNED NOT NULL,
  policy       varchar(32) NOT NULL,
  status       varchar(32) NOT NULL,
  fileid       varchar(32) NOT NULL,   -- same as base id 
  filedata_id  INT UNSIGNED NULL,
  filesize     INT UNSIGNED NULL,
  filechecksum varchar(64) NULL,
  filehaslink  BOOL DEFAULT 0,
  filedate     DATE,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c01    FOREIGN KEY (owner_id)
                               REFERENCES cgnsOwner(id)
                               ON DELETE NO ACTION,
     CONSTRAINT daxctrl_c02    FOREIGN KEY (filedata_id)
                               REFERENCES cgnsBlobEntry(id)
                               ON DELETE NO ACTION
  -- ---------------------------------------------------------
  )
""" % mapping.defaultMapping
# ------------------------------------------------------------
cgnsBlobEntryTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsBlobEntry (
  -- ---------------------------------------------------------
  id           INT UNSIGNED,
  filedata     LONGBLOB NOT NULL,
  entry_id     INT UNSIGNED NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c03    FOREIGN KEY (entry_id)
                               REFERENCES cgnsEntry(id)
                               ON DELETE CASCADE
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsOwnerTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsOwner (
  -- ---------------------------------------------------------
  id           INT UNSIGNED,
  name         varchar(32) DEFAULT '' NOT NULL,
  organisation varchar(32) DEFAULT '' NOT NULL,
  site         varchar(32) DEFAULT '' NOT NULL,
  description  text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id)
  -- ---------------------------------------------------------
  )
""" % mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsLinkTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsLink (
  -- ---------------------------------------------------------
  id           INT UNSIGNED,
  entry_id     INT UNSIGNED NOT NULL,
  linked_id    INT UNSIGNED NULL,
  localpath    varchar(255) DEFAULT '' NOT NULL,
  localnode    varchar(64)  DEFAULT '' NOT NULL,
  linkfile     varchar(64)  DEFAULT '' NOT NULL,
  linknode     varchar(255) DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c04  FOREIGN KEY (entry_id)
                             REFERENCES cgnsEntry(id)
                             ON DELETE CASCADE,
     CONSTRAINT daxctrl_c05  FOREIGN KEY (linked_id)
                             REFERENCES cgnsEntry(id)
                             ON DELETE NO ACTION
  -- ---------------------------------------------------------  
  )
""" % mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsEx = "CREATE UNIQUE INDEX cgnsEntryIX on cgnsEntry(id)"
cgnsExf = "CREATE UNIQUE INDEX cgnsEntryIXf on cgnsEntry(fileid)"
cgnsExfk = "CREATE UNIQUE INDEX cgnsEntryIXfk on cgnsEntry(filedata_id)"
cgnsExfko = "CREATE UNIQUE INDEX cgnsEntryIXfko on cgnsEntry(owner_id)"
cgnsBx = "CREATE UNIQUE INDEX cgnsBlobEntryIX on cgnsBlobEntry(id)"
cgnsBxfke = "CREATE UNIQUE INDEX cgnsBlobEntryIXfke on cgnsBlobEntry(entry_id)"
cgnsOx = "CREATE UNIQUE INDEX cgnsOwnerIX on cgnsOwner(id)"
cgnsLx = "CREATE UNIQUE INDEX cgnsLinkIX on cgnsLink(id)"
cgnsLxfke = "CREATE UNIQUE INDEX cgnsLinkIXfke on cgnsLink(entry_id)"
cgnsLxfkl = "CREATE UNIQUE INDEX cgnsLinkIXfkl on cgnsLink(linked_id)"
# ------------------------------------------------------------
# order is significant...
#
tableList = [
    ['cgnsOwner', cgnsOwnerTable],
    ['cgnsBlobEntry', cgnsBlobEntryTable],
    ['cgnsEntry', cgnsEntryTable],
    ['cgnsLink', cgnsLinkTable],
]
indexList = [
    ['cgnsEntryIX', cgnsEx],
    ['cgnsEntryIXf', cgnsExf],
    ['cgnsEntryIXfk', cgnsExfk],
    ['cgnsEntryIXfko', cgnsExfko],
    ['cgnsBx', cgnsBx],
    ['cgnsBxfke', cgnsBxfke],
    ['cgnsOx', cgnsOx],
    ['cgnsLx', cgnsLx],
    ['cgnsLxfke', cgnsLxfke],
    ['cgnsLxfkl', cgnsLxfkl],
]
#
# ------------------------------------------------------------
