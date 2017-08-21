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
# ------------------------------------------------------------
cgnsPlatformTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPlatform (
  -- ---------------------------------------------------------
  id          INT UNSIGNED,
  nickname    varchar(32),              
  description text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id)
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsPerfMeasureTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPerfMeasure (
  -- ---------------------------------------------------------
  id      INT UNSIGNED,
  time    varchar(32),
  memory  varchar(32),
  -- ---------------------------------------------------------
     PRIMARY KEY(id)
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsPDMDataTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsPDMData (
  -- ---------------------------------------------------------
  id            INT UNSIGNED,
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
     CONSTRAINT daxctrl_c11 FOREIGN KEY (entry_id)
                            REFERENCES cgnsEntry(id)
                            ON DELETE CASCADE
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsSystemTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsSystem (
  -- ---------------------------------------------------------
  id               INT UNSIGNED,
  entry_id         INT UNSIGNED NULL,
  creationdate     DATE,
  modificationdate DATE,
  platform_id      INT UNSIGNED NULL,
  perfmeasure_id   INT UNSIGNED NULL,
  pdm_id           INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
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
cgnsTestTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsTest (
  -- ---------------------------------------------------------
  id            INT UNSIGNED,
  entry_id      INT UNSIGNED NULL,
  number        varchar(32) NOT NULL,
  geometry      varchar(32) NOT NULL,
  family_id     INT UNSIGNED NULL,
  remarks       text DEFAULT '' NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY(id),
     CONSTRAINT daxctrl_c16 FOREIGN KEY (entry_id)
                            REFERENCES cgnsEntry(id)
                            ON DELETE CASCADE,
     CONSTRAINT daxctrl_c17 FOREIGN KEY (family_id)
                            REFERENCES cgnsTestFamily(id)
                            ON DELETE RESTRICT
  -- ---------------------------------------------------------
  )
"""
cgnsTestFamilyTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsTestFamily (
  -- ---------------------------------------------------------
  id           INT UNSIGNED
               PRIMARY KEY,
  name         varchar(32) NOT NULL,
  description  text DEFAULT '' NOT NULL
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsLogTable = """
  -- ---------------------------------------------------------
  CREATE TABLE cgnsLog (
  -- ---------------------------------------------------------
  id           INT UNSIGNED
               PRIMARY KEY,
  connection   varchar(32) NOT NULL,
  stamp        DATETIME,
  log          text
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsPTx = "CREATE UNIQUE INDEX cgnsPlatformIX on cgnsPlatform (id)"
cgnsPMx = "CREATE UNIQUE INDEX cgnsPerfMeasureIX on cgnsPerfMeasure(id)"
cgnsPDx = "CREATE UNIQUE INDEX cgnsPDMDataIX on cgnsPDMData (id)"
cgnsPDxfke = "CREATE UNIQUE INDEX cgnsPDMDataIXfke on cgnsPDMData(entry_id)"
cgnsSx = "CREATE UNIQUE INDEX cgnsSystemIX on cgnsSystem (id)"
cgnsSxfke = "CREATE UNIQUE INDEX cgnsSystemIXfke on cgnsSystem (entry_id)"
cgnsSxfkp = "CREATE UNIQUE INDEX cgnsSystemIXfkp on cgnsSystem (platform_id)"
cgnsSxfkm = "CREATE UNIQUE INDEX cgnsSystemIXfkm on cgnsSystem (perfmeasure_id)"
cgnsSxfkv = "CREATE UNIQUE INDEX cgnsSystemIXfkv on cgnsSystem (pdm_id)"
cgnsTx = "CREATE UNIQUE INDEX cgnsTestIX on cgnsTest (id)"
cgnsTxfke = "CREATE UNIQUE INDEX cgnsTestIXfke on cgnsTest (entry_id)"
cgnsTxfkf = "CREATE UNIQUE INDEX cgnsTestIXfkf on cgnsTest (family_id)"
# ------------------------------------------------------------
# Table list is a dictionnary ;)
tableList = [
    ['cgnsPlatform', cgnsPlatformTable],
    ['cgnsTest', cgnsTestTable],
    ['cgnsTestFamily', cgnsTestFamilyTable],
    ['cgnsPerfMeasure', cgnsPerfMeasureTable],
    ['cgnsPDMData', cgnsPDMDataTable],
    ['cgnsSystem', cgnsSystemTable],
    ['cgnsLog', cgnsLogTable],
]
indexList = [
    ['cgnsPTx', cgnsPTx],
    ['cgnsPMx', cgnsPMx],
    ['cgnsPDx', cgnsPDx],
    ['cgnsPDxfke', cgnsPDxfke],
    ['cgnsSx', cgnsSx],
    ['cgnsSxfke', cgnsSxfke],
    ['cgnsSxfkp', cgnsSxfkp],
    ['cgnsSxfkm', cgnsSxfkm],
    ['cgnsSxfkv', cgnsSxfkv],
    ['cgnsTx', cgnsTx],
    ['cgnsTxfke', cgnsTxfke],
    ['cgnsTxfkf', cgnsTxfkf],
]
#
# ------------------------------------------------------------
