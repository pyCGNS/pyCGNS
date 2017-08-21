#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System - 
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# ------------------------------------------------------------
# pyDAX - DBMS schema - foreign keys ids
# ------------------------------------------------------------
#
# This file can be configured for your own existing DBMS
#
# THESE ARE STRING SUBSTITUTIONS
# - FK stands for foreign key
# - EN stands for enumerates 
#
defaultMapping = {
    'EN_FILESTATUS': '"UNKNOWN","DRAFT","BASELINE","REFERENCE","OBSOLETE"',
    'EN_FILEPOLICY': '"UNKNOWN","NONE","PRIVATE","CHANCE"',
    'FK_CGNSOWNER': 'cgnsOwner(id)',
    'FK_TESTFAMILY': 'cgnsTestFamily(family_id)',
    'FK_PLATFORM': 'cgnsPlatform(platform_id)',
    'FK_PDMDATA': 'cgnsPDMData(data_id)',
    'FK_PERFMEASURE': 'cgnsPerfMeasure(perf_id)',
    'FK_TEST': 'cgnsTest(test_id)',
}
#
PUBLIC = 'NONE'
PRIVATE = 'PRIVATE'
CHANCE = 'CHANCE'
#
UNKNOWN = 'UNKNOWN'
DRAFT = 'DRAFT'
BASELINE = 'BASELINE'
REFERENCE = 'REFERENCE'
OBSOLETE = 'OBSOLETE'
#
FileStatusEnum = eval(defaultMapping['EN_FILESTATUS'])
#
