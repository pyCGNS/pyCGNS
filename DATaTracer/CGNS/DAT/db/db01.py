#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# ------------------------------------------------------------
# pyDAX - DBMS schema - Basic data
# ------------------------------------------------------------
#
# This file can be configured for your own existing DBMS
# Some attribute names can be changed (foreign keys)
# see the foreignKeys.py file
#
import foreignKeys as mapping
#
# AUTO_INCREMENT 
#
# ------------------------------------------------------------
cgnsEntryTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsEntry (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL AUTO_INCREMENT,
  owner_id     INT UNSIGNED NOT NULL,
  policy       ENUM (%(EN_FILEPOLICY)s) NOT NULL,
  status       ENUM (%(EN_FILESTATUS)s) NOT NULL,
  fileid       varchar(32) NOT NULL,   -- same as base id 
  filedata_id  INT UNSIGNED NULL,
  filesize     INT UNSIGNED NULL,
  filechecksum varchar(64) NULL,
  filehaslink  BOOL DEFAULT 0,
  filedate     DATE,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     UNIQUE INDEX cgnsEntryIX    (id),
     UNIQUE INDEX cgnsEntryIXf   (fileid),     
     UNIQUE INDEX cgnsEntryIXfk  (filedata_id),
     INDEX cgnsEntryIXfko (owner_id),
     CONSTRAINT daxctrl_c01    FOREIGN KEY (owner_id)
                               REFERENCES cgnsOwner(id)
                               ON DELETE NO ACTION,
     CONSTRAINT daxctrl_c02    FOREIGN KEY (filedata_id)
                               REFERENCES cgnsBlobEntry(id)
                               ON DELETE NO ACTION
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
# ------------------------------------------------------------
cgnsBlobEntryTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsBlobEntry (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL AUTO_INCREMENT,
  filedata     LONGBLOB NOT NULL,
  entry_id     INT UNSIGNED NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     UNIQUE INDEX cgnsBlobEntryIX   (id),
     UNIQUE INDEX cgnsBlobEntryIXfke(entry_id)
  -- ---------------------------------------------------------
  )
"""
cgnsBlobEntryForeignKey="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsBlobEntry ADD
     CONSTRAINT daxctrl_c03    FOREIGN KEY (entry_id)
                               REFERENCES cgnsEntry(id)
                               ON DELETE CASCADE
  -- ---------------------------------------------------------
"""  
# ------------------------------------------------------------
cgnsOwnerTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsOwner (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name         varchar(32) DEFAULT '' NOT NULL,
  organisation varchar(32) DEFAULT '' NOT NULL,
  site         varchar(32) DEFAULT '' NOT NULL,
  description  text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsOwnerIX (id)
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsLinkTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsLink (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL AUTO_INCREMENT,
  entry_id     INT UNSIGNED NOT NULL,
  linked_id    INT UNSIGNED NULL,
  localpath    varchar(255) DEFAULT '' NOT NULL,
  localnode    varchar(64)  DEFAULT '' NOT NULL,
  linkfile     varchar(64)  DEFAULT '' NOT NULL,
  linknode     varchar(255) DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsLinkIX    (id),
     INDEX cgnsLinkIXfke (entry_id),
     INDEX cgnsLinkIXfkl (linked_id),
     CONSTRAINT daxctrl_c04  FOREIGN KEY (entry_id)
                             REFERENCES cgnsEntry(id)
                             ON DELETE CASCADE,
     CONSTRAINT daxctrl_c05  FOREIGN KEY (linked_id)
                             REFERENCES cgnsEntry(id)
                             ON DELETE NO ACTION
  -- ---------------------------------------------------------  
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
# order is significant...
#
tableList=[
  ['cgnsOwner',               cgnsOwnerTable],
  ['cgnsBlobEntry',           cgnsBlobEntryTable],  
  ['cgnsEntry',               cgnsEntryTable],
  ['cgnsLink' ,               cgnsLinkTable],
  ['cgnsBlobEntryForeignKey', cgnsBlobEntryForeignKey],    
]
#
# ------------------------------------------------------------
