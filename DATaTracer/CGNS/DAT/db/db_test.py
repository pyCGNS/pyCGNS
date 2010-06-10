#  -------------------------------------------------------------------------
#  pyCGNS.DAT - Python package for CFD General Notation System - DATaTracer
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release$
#  -------------------------------------------------------------------------
# ------------------------------------------------------------
# pyDAX - DBMS schema - values
# ------------------------------------------------------------
#
defaultTestFamily=[
  """INSERT \
     INTO cgnsTestFamily(name,description)\
     values (%s,%s)""",
  (
    ("Fuselage","Isolated Fuselage"),
    ("Hover",   "Isolated Rotor in Hover"),
    ("Forward", "Isolated Rotor in Forward Flight"),
    ("Actuator","Fuselage and Actuator disk"),
    ("Complete","Complete Helicopter"),    
  )
]
#
defaultTest=[
  """INSERT \
     INTO cgnsTest(number,geometry,family_id,remarks)\
     select %s,%s,id,%s\
     from cgnsTestFamily\
     where name=%s""",
  (
    ("1.1-F1", "HELIFUSE TC1","Geometry C1","Fuselage"),
    ("1.1-F2", "HELIFUSE TC3","Geometry C1","Fuselage"),
    ("1.1-F3", "HELIFUSE TC4","Geometry C2","Fuselage"),
    ("1.1-F4", "HELIFUSE TC5","Geometry C2","Fuselage"),
    ("1.1-F5", "DGV ONERA test","Configuration C4","Fuselage"),
    ("1.1-F6", "PROSEAM NH90 Air intake","0 km/h; 5kg/s","Fuselage"),
    ("1.1-F7", "PROSEAM NH90 Air intake","300 km/h; kg/s","Fuselage"),
    ("1.1-F8", "HELIFLOW Task 1, #44","$\mu$=0.05","Fuselage"),
    ("1.1-F10","NH90 or DTV Fuselage","Model","Fuselage"),
    ("2.1-H1", "7A-Rotor, Marignane","-","Hover"),
    ("2.1-H2", "HELISHAPE, 7A-Rotor, #313","-","Hover"),
    ("2.1-H3", "7A-Rotor, Marignane","-","Hover"),
    ("2.1-H4", "7AD-Rotor, Marignane","-","Hover"),
    ("2.1-H5", "HELISHAPE, 7AD-Rotor, #279","-","Hover"),        
    ("2.2-FF1","7A-Rotor, Modane, #312","High Speed","Forward"),        
    ("2.2-FF2","7A-Rotor, Modane, #317","High Speed","Forward"),        
    ("2.2-FF3","7A-Rotor, Modane, #321","High Speed","Forward"),        
    ("2.2-FF4","7A-Rotor, Modane, #1224","High Speed","Forward"),        
    ("2.2-FF5","7A-Rotor, Modane, #1228","High Speed","Forward"),        
    ("1.2-A1", "Dauphin, S2Ch","F +M/R","Actuator"),
    ("1.2-A2", "HELIFLOW Task 1, #13","F +M/R (Zero flap)","Actuator"),
    ("1.2-A4", "HELIFLOW Task 5, #6, BO105","F +M/R","Actuator"),
    ("1.2-A5", "HELIFLOW Task 3, Test run 1","F +M/R +T/R","Actuator"),
    ("1.2-A6", "NH90, DNW, Recirculation","F +M/R +Jet","Actuator"),
    ("3-C1",   "Dauphin, S2Ch","F +M/R","Complete"),
    ("3-C2",   "HELIFLOW Task 1, #13","F +M/R","Complete"),
    ("3-C3",   "HELIFLOW Task 3, #1, BO105","F +M/R +T/R","Complete"),
    ("3-C4",   "HELIFLOW Task 3, #32, BO105","F +M/R +T/R","Complete"),
    ("3-C5",   "HELIFLOW Task 3, #37, BO105","F +M/R +T/R","Complete"),
    ("3-C11",  "DTV (or NH90), Quartering Flight","F +M/R +T/R","Complete"),
    ("3-C12",  "DTV (or NH90), Air Intake","F +M/R +T/R","Complete"),
    ("3-C13",  "DTV (or NH90), Pitch-Up","F +M/R +T/R","Complete"),
    ("3-C14",  "DTV (or NH90), High g","F +M/R +T/R","Complete"),
    ("3-C15",  "DTV (or NH90), High V","F +M/R +T/R","Complete"),
  )
]    
#
valuesList=[
 ['cgnsTestFamilyList', defaultTestFamily],
 ['cgnsTestList', defaultTest],
]
