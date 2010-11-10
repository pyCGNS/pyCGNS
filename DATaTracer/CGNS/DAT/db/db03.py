#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# ------------------------------------------------------------
# pyDAX - DBMS schema - CGNS data
# ------------------------------------------------------------
#
import foreignKeys as mapping
#
# ------------------------------------------------------------
cgnsBaseInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsBaseInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED,
  entry_id       INT UNSIGNED NULL,
  name           varchar(32) NOT NULL,
  title          varchar(64) NOT NULL,
  description    text DEFAULT '' NOT NULL,
  remarks        text DEFAULT '' NOT NULL,
  physicaldim    INT UNSIGNED NOT NULL,
  celldim        INT UNSIGNED NOT NULL,
  nzones         INT UNSIGNED NOT NULL,
  simulation     varchar(32) NOT NULL,
  reference_id   INT UNSIGNED NULL,
  floweq_id      INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c21  FOREIGN KEY (entry_id)
                             REFERENCES cgnsEntry(id)
                             ON DELETE CASCADE,
     CONSTRAINT daxctrl_c22  FOREIGN KEY (reference_id)
                             REFERENCES cgnsReferenceInfo(id)
                             ON DELETE NO ACTION,
     CONSTRAINT daxctrl_c23  FOREIGN KEY (floweq_id)
                             REFERENCES cgnsFlowEquationSetInfo(id)
                             ON DELETE NO ACTION
  -- ---------------------------------------------------------
 )
"""
cgnsReferenceInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsReferenceInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED,
  base_id        INT UNSIGNED NULL,
  attlist_id     INT UNSIGNED NULL,
  ReferenceStateDescription   varchar(32) NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c26  FOREIGN KEY (attlist_id)
                             REFERENCES cgnsAttributeList(id)
                             ON DELETE NO ACTION
     CONSTRAINT daxctrl_c25  FOREIGN KEY (base_id)
                             REFERENCES cgnsBaseInfo(id)
                             ON DELETE CASCADE
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsFlowEquationSetInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowEquationSetInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED,
  base_id        INT UNSIGNED NULL,
  attlist_id     INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c27  FOREIGN KEY (attlist_id)
                             REFERENCES cgnsAttributeList(id)
                             ON DELETE NO ACTION
     CONSTRAINT daxctrl_c28  FOREIGN KEY (base_id)
                             REFERENCES cgnsBaseInfo(id)
                             ON DELETE CASCADE
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsAttributeListTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsAttributeList (
  -- ---------------------------------------------------------
  id             INT UNSIGNED,
  path           varchar(64),
  modified       BOOL,
  entry_id       INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c30       FOREIGN KEY (entry_id)
                                  REFERENCES cgnsEntry(id)
                                  ON DELETE CASCADE
  -- ---------------------------------------------------------  
  )
"""
# ------------------------------------------------------------
cgnsAttributeTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsAttribute (
  -- ---------------------------------------------------------
  id             INT UNSIGNED,
  alist_id       INT UNSIGNED NULL,
  path           varchar(64),
  a_value        varchar(64),
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     CONSTRAINT daxctrl_c29  FOREIGN KEY (alist_id)
                             REFERENCES cgnsAttributeList(id)
                             ON DELETE CASCADE
  -- ---------------------------------------------------------
  )
"""
# ------------------------------------------------------------
cgnsZoneInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsZoneInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED PRIMARY KEY,
  entry_id       INT UNSIGNED NOT NULL
                 REFERENCES cgnsEntry(entry_id),
  name           varchar(32) NOT NULL,
  idim           INT UNSIGNED NOT NULL,
  jdim           INT UNSIGNED NOT NULL,
  kdim           INT UNSIGNED NOT NULL,
  nflowsolutions INT UNSIGNED NOT NULL
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
# ------------------------------------------------------------
cgnsExperimentalDataTable="""
  -- ---------------------------------------------------------
  -- NOT USED YET
  -- ---------------------------------------------------------  
  CREATE TABLE cgnsExperimentalData (
  -- ---------------------------------------------------------
  id             INT UNSIGNED PRIMARY KEY,
  test_id        INT UNSIGNED NOT NULL
                 REFERENCES %(FK_TEST)s
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsSolverInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsSolverInfo (
  -- ---------------------------------------------------------
  id               INT UNSIGNED PRIMARY KEY,
  entry_id         INT UNSIGNED NOT NULL
                   REFERENCES cgnsEntry(id),
  solverversion    varchar(32) NOT NULL,
  solvername       varchar(32) NOT NULL,
--  multigrid        BOOL DEFAULT 0,
--  dissipation      ENUM ('scalar','matrix'),
--  timeresolution   ENUM ('RungeKutta','BackwardEuler','DTS'),
--  spaceresolution  ENUM ('IRS','LU'),
--  schema           ENUM ('Jameson','Van Leer','Roe'),
--  multigridcycle   varchar(32)
  nbparallel       INT UNSIGNED DEFAULT 1
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsFlowSolutionInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowSolutionInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED PRIMARY KEY,
  entry_id       INT UNSIGNED NOT NULL
                 REFERENCES cgnsEntry(id),
  zone_id        INT UNSIGNED NOT NULL
                 REFERENCES cgnsZoneInfoTable(id),
--  gridlocation   ENUM ('CellCenter','Vertex')
--                 NOT NULL DEFAULT 'CellCenter',
  name           varchar(32) NOT NULL
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsFlowSolutionFieldInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowSolutionFieldInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED PRIMARY KEY,
  solution_id    INT UNSIGNED NOT NULL
                 REFERENCES cgnsFlowSolutionInfo(id),
  name           varchar(32) NOT NULL,
  datatype       varchar(2) NOT NULL,
  remark         text DEFAULT '' NOT NULL
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
# ------------------------------------------------------------
cgnsBx="CREATE UNIQUE INDEX cgnsBaseInfoIX on cgnsBaseInfo(id)"
cgnsBxfke="CREATE UNIQUE INDEX cgnsBaseInfoIXfke on cgnsBaseInfo(entry_id)"
cgnsBxfkr="CREATE UNIQUE INDEX cgnsBaseInfoIXfkr on cgnsBaseInfo(reference_id)"
cgnsBxfkf="CREATE UNIQUE INDEX cgnsBaseInfoIXfkf on cgnsBaseInfo(floweq_id)"
cgnsRx="CREATE UNIQUE INDEX cgnsReferenceInfoIX on cgnsReferenceInfo (id)"
cgnsRxb="CREATE UNIQUE INDEX cgnsReferenceInfoIXb on cgnsReferenceInfo(base_id)"
cgnsRxa="CREATE UNIQUE INDEX cgnsReferenceInfoIXa on cgnsReferenceInfo(attlist_id)"
cgnsFx="CREATE UNIQUE INDEX cgnsFlowEquationSetInfoIX on cgnsFlowEquationSetInfo (id)"
cgnsFxb="CREATE UNIQUE INDEX cgnsFlowEquationSetInfoIXb on cgnsFlowEquationSetInfo (base_id)"
cgnsFxa="CREATE UNIQUE INDEX cgnsFlowEquationSetInfoIXa on cgnsFlowEquationSetInfo (attlist_id)"
cgnsALx="CREATE UNIQUE INDEX cgnsAttributeListIX on cgnsAttributeList(id)"
cgnsALxp="CREATE UNIQUE INDEX cgnsAttributeListIXp on cgnsAttributeList (path)"
cgnsALxfke="CREATE UNIQUE INDEX cgnsAttributeListIXfke on cgnsAttributeList(entry_id)"
cgnsAx="CREATE UNIQUE INDEX cgnsAttributeIX on cgnsAttribute (id)"
cgnsAxfk="CREATE UNIQUE INDEX cgnsAttributeIXfk on cgnsAttribute(alist_id)"
cgnsZx="CREATE UNIQUE INDEX cgnsZoneInfoIX on cgnsZoneInfo (id,entry_id,name)"
cgnsEx="CREATE UNIQUE INDEX cgnsExperimentalDataIX on cgnsExperimentalData (id,test_id)"
cgnsSx="CREATE UNIQUE INDEX cgnsSolverInfoIX on cgnsSolverInfo (id,entry_id,solvername)"
cgnsFSx="CREATE UNIQUE INDEX cgnsFlowSolutionInfoIX on cgnsFlowSolutionInfo (id,entry_id,zone_id,name)"
cgnsFFx="CREATE UNIQUE INDEX cgnsFlowSolutionFieldInfoIX on cgnsFlowSolutionFieldInfo (id,solution_id,name)"
# ------------------------------------------------------------
tableList=[
  ['cgnsExperimentalData',           cgnsExperimentalDataTable],
  ['cgnsSolverInfo',                 cgnsSolverInfoTable],
  ['cgnsFlowEquationSetInfo',        cgnsFlowEquationSetInfoTable],  
  ['cgnsZoneInfo',                   cgnsZoneInfoTable],
  ['cgnsFlowSolutionInfo',           cgnsFlowSolutionInfoTable],
  ['cgnsFlowSolutionFieldInfo',      cgnsFlowSolutionFieldInfoTable],
  ['cgnsAttributeList',              cgnsAttributeListTable],
  ['cgnsAttribute',                  cgnsAttributeTable],      
  ['cgnsReferenceInfo',              cgnsReferenceInfoTable],  
  ['cgnsBaseInfo',                   cgnsBaseInfoTable],
]
indexList=[
  ['cgnsBx', cgnsBx],
  ['cgnsBxfke', cgnsBxfke],
  ['cgnsBxfkr', cgnsBxfkr],
  ['cgnsBxfkf', cgnsBxfkf],
  ['cgnsRx', cgnsRx],
  ['cgnsRxa', cgnsRxa],
  ['cgnsRxb', cgnsRxb],
  ['cgnsFx', cgnsFx],
  ['cgnsFxb', cgnsFxb],
  ['cgnsFxa', cgnsFxa],
  ['cgnsALx', cgnsALx],
  ['cgnsALxp', cgnsALxp],
  ['cgnsALxfke', cgnsALxfke],
  ['cgnsAx', cgnsAx],
  ['cgnsAxfk', cgnsAxfk],
  ['cgnsZx', cgnsZx],
  ['cgnsEx', cgnsEx],
  ['cgnsSx', cgnsSx],
  ['cgnsFSx', cgnsFSx],
  ['cgnsFFx', cgnsFFx],
]
#
# ------------------------------------------------------------
