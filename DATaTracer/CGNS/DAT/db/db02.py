#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# ------------------------------------------------------------
# pyDAX - DBMS schema - Meta data
# ------------------------------------------------------------
#
# This file can be configured for your own existing DBMS
# Some attribute names can be changed (foreign keys)
# see the foreignKeys.py file
#
import foreignKeys as mapping
#
# ------------------------------------------------------------
cgnsPlatformTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPlatform (
  -- ---------------------------------------------------------
  id          INT UNSIGNED NOT NULL AUTO_INCREMENT,
  nickname    varchar(32),              
  description text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsPlatformIX (id)
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsPerfMeasureTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPerfMeasure (
  -- ---------------------------------------------------------
  id      INT UNSIGNED NOT NULL AUTO_INCREMENT,
  time    varchar(32),
  memory  varchar(32),
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
     INDEX cgnsPerfMeasureIX (id)
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsPDMDataTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPDMData (
  -- ---------------------------------------------------------
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  entry_id      INT UNSIGNED NULL,
  fileversion   INT UNSIGNED,
  filerelease   INT UNSIGNED,
  filechange    varchar(16),
  checkin_ct    INT UNSIGNED DEFAULT 0,  
  checkout_ct   INT UNSIGNED DEFAULT 0,  
  update_ct     INT UNSIGNED DEFAULT 0,  
  import_ct     INT UNSIGNED DEFAULT 0,  
  export_ct     INT UNSIGNED DEFAULT 0,  
  modified      BOOL,
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
     INDEX cgnsPDMDataIX    (id),
     INDEX cgnsPDMDataIXfke (entry_id),
     CONSTRAINT daxctrl_c11 FOREIGN KEY (entry_id)
                            REFERENCES cgnsEntry(id)
                            ON DELETE CASCADE,
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsSystemTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsSystem (
  -- ---------------------------------------------------------
  id               INT UNSIGNED NOT NULL AUTO_INCREMENT,
  entry_id         INT UNSIGNED NULL,
  creationdate     DATE,
  modificationdate DATE,
  platform_id      INT UNSIGNED NULL,
  perfmeasure_id   INT UNSIGNED NULL,
  pdm_id           INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
     INDEX cgnsSystemIX     (id),
     INDEX cgnsSystemIXfke  (entry_id),
     INDEX cgnsSystemIXfkp  (platform_id),
     INDEX cgnsSystemIXfkm  (perfmeasure_id),
     INDEX cgnsSystemIXfkv  (pdm_id),
     CONSTRAINT daxctrl_c12 FOREIGN KEY (entry_id)
                            REFERENCES cgnsEntry(id)
                            ON DELETE CASCADE,
     CONSTRAINT daxctrl_c13 FOREIGN KEY (platform_id)
                            REFERENCES cgnsPlatform(id)
                            ON DELETE RESTRICT,
     CONSTRAINT daxctrl_c14 FOREIGN KEY (perfmeasure_id)
                            REFERENCES cgnsPerfMeasure(id)
                            ON DELETE NO ACTION,
     CONSTRAINT daxctrl_c15 FOREIGN KEY (pdm_id)
                            REFERENCES cgnsPDMData(id)
                            ON DELETE NO ACTION
  -- ---------------------------------------------------------
  )
"""
#
# ------------------------------------------------------------
cgnsTestTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsTest (
  -- ---------------------------------------------------------
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  entry_id      INT UNSIGNED NULL,
  number        varchar(32) NOT NULL,
  geometry      varchar(32) NOT NULL,
  family_id     INT UNSIGNED NULL,
  remarks       text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
     INDEX cgnsTestIX       (id),
     INDEX cgnsTestIXfke    (entry_id),
     INDEX cgnsTestIXfkf    (family_id)
  -- ---------------------------------------------------------
  )
"""
cgnsTestForeignKey1="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsTest ADD
     CONSTRAINT daxctrl_c16 FOREIGN KEY (entry_id)
                            REFERENCES cgnsEntry(id)
                            ON DELETE CASCADE
  -- ---------------------------------------------------------
"""
cgnsTestForeignKey2="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsTest ADD
     CONSTRAINT daxctrl_c17 FOREIGN KEY (family_id)
                            REFERENCES cgnsTestFamily(id)
                            ON DELETE RESTRICT
  -- ---------------------------------------------------------
"""
# ------------------------------------------------------------
cgnsTestFamilyTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsTestFamily (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL
               AUTO_INCREMENT PRIMARY KEY,
  name         varchar(32) NOT NULL,
  description  text DEFAULT '' NOT NULL
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsLogTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsLog (
  -- ---------------------------------------------------------
  id           INT UNSIGNED NOT NULL
               AUTO_INCREMENT PRIMARY KEY,
  connection   varchar(32) NOT NULL,
  stamp        DATETIME,
  log          text
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
# Table list is a dictionnary ;)
tableList=[
  ['cgnsPlatform',         cgnsPlatformTable],
  ['cgnsTest',             cgnsTestTable],
  ['cgnsTestFamily',       cgnsTestFamilyTable],
  ['cgnsPerfMeasure',      cgnsPerfMeasureTable],    
  ['cgnsPDMData',          cgnsPDMDataTable],    
  ['cgnsSystem',           cgnsSystemTable],
  ['cgnsLog',              cgnsLogTable],      
  ['cgnsTestForeignKey1',  cgnsTestForeignKey1],    
  ['cgnsTestForeignKey2',  cgnsTestForeignKey2],    
]
#
# ------------------------------------------------------------
