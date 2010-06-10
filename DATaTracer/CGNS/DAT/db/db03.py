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
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
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
     INDEX cgnsBaseInfoIX    (id),
     INDEX cgnsBaseInfoIXfke (entry_id),
     INDEX cgnsBaseInfoIXfkr (reference_id),
     INDEX cgnsBaseInfoIXfkf (floweq_id),     
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
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
  base_id        INT UNSIGNED NULL,
  attlist_id     INT UNSIGNED NULL,
  ReferenceStateDescription   varchar(32) NOT NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsReferenceInfoIX    (id),
     INDEX cgnsReferenceInfoIXb   (base_id),
     INDEX cgnsReferenceInfoIXa   (attlist_id)          
  -- ---------------------------------------------------------
  )
"""
cgnsReferenceInfoForeignKey1="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsReferenceInfo ADD
     CONSTRAINT daxctrl_c26  FOREIGN KEY (attlist_id)
                             REFERENCES cgnsAttributeList(id)
                             ON DELETE NO ACTION
  -- ---------------------------------------------------------
"""
cgnsReferenceInfoForeignKey2="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsReferenceInfo ADD
     CONSTRAINT daxctrl_c25  FOREIGN KEY (base_id)
                             REFERENCES cgnsBaseInfo(id)
                             ON DELETE CASCADE
  -- ---------------------------------------------------------
"""
# ------------------------------------------------------------
cgnsFlowEquationSetInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowEquationSetInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
  base_id        INT UNSIGNED NULL,
  attlist_id     INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsFlowEquationSetInfoIX   (id),
     INDEX cgnsFlowEquationSetInfoIXb  (base_id),
     INDEX cgnsFlowEquationSetInfoIXa  (attlist_id)          
  -- ---------------------------------------------------------
  )
"""
cgnsFlowEquationSetInfoForeignKey1="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsFlowEquationSetInfo ADD
     CONSTRAINT daxctrl_c27  FOREIGN KEY (attlist_id)
                             REFERENCES cgnsAttributeList(id)
                             ON DELETE NO ACTION
  -- ---------------------------------------------------------
"""
cgnsFlowEquationSetInfoForeignKey2="""
  -- ---------------------------------------------------------
     ALTER TABLE cgnsFlowEquationSetInfo ADD
     CONSTRAINT daxctrl_c28  FOREIGN KEY (base_id)
                             REFERENCES cgnsBaseInfo(id)
                             ON DELETE CASCADE
  -- ---------------------------------------------------------
"""
# ------------------------------------------------------------
cgnsAttributeListTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsAttributeList (
  -- ---------------------------------------------------------
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
  path           varchar(64),
  modified       BOOL,
  entry_id       INT UNSIGNED NULL,
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsAttributeListIX    (id),
     UNIQUE INDEX cgnsAttributeListIXp   (path),
     INDEX cgnsAttributeListIXfke (entry_id),
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
  id             INT UNSIGNED NOT NULL AUTO_INCREMENT,
  alist_id       INT UNSIGNED NULL,
  path           varchar(64),
  a_value        varchar(64),
  -- ---------------------------------------------------------
     PRIMARY KEY (id),
     INDEX cgnsAttributeIX   (id),
     INDEX cgnsAttributeIXfk (alist_id),
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
  id             INT UNSIGNED NOT NULL
                 AUTO_INCREMENT PRIMARY KEY,
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
#
cgnsZoneInfoIndex="""
  CREATE INDEX cgnsZoneInfoIndex
  ON cgnsZoneInfo (id,entry_id,name)
"""
# ------------------------------------------------------------
cgnsExperimentalDataTable="""
  -- ---------------------------------------------------------
  -- NOT USED YET
  -- ---------------------------------------------------------  
  CREATE TABLE cgnsExperimentalData (
  -- ---------------------------------------------------------
  id             INT UNSIGNED NOT NULL
                 AUTO_INCREMENT PRIMARY KEY,
  test_id        INT UNSIGNED NOT NULL
                 REFERENCES %(FK_TEST)s
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
cgnsExperimentalDataIndex="""
  CREATE INDEX cgnsExperimentalData
  ON cgnsExperimentalData (id,test_id)
"""
# ------------------------------------------------------------
cgnsSolverInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsSolverInfo (
  -- ---------------------------------------------------------
  id               INT UNSIGNED NOT NULL
                   AUTO_INCREMENT PRIMARY KEY,
  entry_id         INT UNSIGNED NOT NULL
                   REFERENCES cgnsEntry(id),
  solverversion    varchar(32) NOT NULL,
  solvername       varchar(32) NOT NULL,
  nbparallel       INT UNSIGNED DEFAULT 1,
  multigrid        BOOL DEFAULT 0,
  dissipation      ENUM ('scalar','matrix'),
  timeresolution   ENUM ('RungeKutta','BackwardEuler','DTS'),
  spaceresolution  ENUM ('IRS','LU'),
  schema           ENUM ('Jameson','Van Leer','Roe'),
  multigridcycle   varchar(32)
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
cgnsSolverInfoIndex="""
  CREATE INDEX cgnsSolverInfoIndex
  ON cgnsSolverInfo (id,entry_id,solvername)
"""
# ------------------------------------------------------------
cgnsFlowSolutionInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowSolutionInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED NOT NULL
                 AUTO_INCREMENT PRIMARY KEY,
  entry_id       INT UNSIGNED NOT NULL
                 REFERENCES cgnsEntry(id),
  zone_id        INT UNSIGNED NOT NULL
                 REFERENCES cgnsZoneInfoTable(id),
  gridlocation   ENUM ('CellCenter','Vertex')
                 NOT NULL DEFAULT 'CellCenter',
  name           varchar(32) NOT NULL
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
cgnsFlowSolutionInfoIndex="""
  CREATE INDEX cgnsFlowSolutionInfoIndex
  ON cgnsFlowSolutionInfo (id,entry_id,zone_id,name)
"""
# ------------------------------------------------------------
cgnsFlowSolutionFieldInfoTable="""
  -- ---------------------------------------------------------
  CREATE TABLE cgnsFlowSolutionFieldInfo (
  -- ---------------------------------------------------------
  id             INT UNSIGNED NOT NULL
                 AUTO_INCREMENT PRIMARY KEY,
  solution_id    INT UNSIGNED NOT NULL
                 REFERENCES cgnsFlowSolutionInfo(id),
  name           varchar(32) NOT NULL,
  datatype       varchar(2) NOT NULL,
  remark         text DEFAULT '' NOT NULL
  -- ---------------------------------------------------------
  )
"""%mapping.defaultMapping
#
cgnsFlowSolutionFieldInfoIndex="""
  CREATE INDEX cgnsFlowSolutionFieldInfoIndex
  ON cgnsFlowSolutionFieldInfo (id,solution_id,name)
"""
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
  ['cgnsReferenceInfoForeignKey1',   cgnsReferenceInfoForeignKey1],
  ['cgnsReferenceInfoForeignKey2',   cgnsReferenceInfoForeignKey2],
]
#
# ------------------------------------------------------------
