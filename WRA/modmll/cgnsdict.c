/* ====================================================================== */
/* CFD General Notation System - CGNS lib wrapper */
/* ONERA/DSNA/ELSA - poinot@onera.fr */
/* pyCGNS - $Rev: 73 $ $Date: 2009-02-06 17:17:44 +0100 (Fri, 06 Feb 2009) $ */
/* See file COPYING in the root directory of this Python module source */
/* tree for license information. */
/* ====================================================================== */

/* DbMIDLEVEL objects */

/* KEYWORDS are now in cgnskeywords.py file */

#include "Python.h"
#include "cgnslib.h"
#include "cgnsstrings.h"

#ifndef CG_MODE_READ
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *\
 *      modes for cgns file                                              *
\* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#define CG_MODE_READ	0
#define CG_MODE_WRITE	1
#define CG_MODE_CLOSED	2
#define CG_MODE_MODIFY	3

/* file types */

#define CG_FILE_NONE 0
#define CG_FILE_ADF  1
#define CG_FILE_HDF5 2
#define CG_FILE_XML  3

/* function return codes */

#define CG_OK		  0
#define CG_ERROR	  1
#define CG_NODE_NOT_FOUND 2
#define CG_INCORRECT_PATH 3
#define CG_NO_INDEX_DIM   4

/* Null and UserDefined enums */

#define CG_Null        0
#define CG_UserDefined 1

/* max goto depth */

#define CG_MAX_GOTO_DEPTH 20

/* configuration options */

#define CG_CONFIG_ERROR     1
#define CG_CONFIG_COMPRESS  2
#define CG_CONFIG_SET_PATH  3
#define CG_CONFIG_ADD_PATH  4
#define CG_CONFIG_FILE_TYPE 5

#define CG_CONFIG_XML_DELETED     301
#define CG_CONFIG_XML_NAMESPACE   302
#define CG_CONFIG_XML_THRESHOLD   303
#define CG_CONFIG_XML_COMPRESSION 304

/* legacy code support */

#define MODE_READ	CG_MODE_READ
#define MODE_WRITE	CG_MODE_WRITE
#define MODE_MODIFY	2
#define Null            CG_Null
#define UserDefined	CG_UserDefined

#endif

/* macros for heavy use of Python dicts ;) */
/* PyDict_SetItemString(xd, xdn##"_", xdd); \ */

#define createDict(xd,xdn,xdd,xddr) \
xdd = PyDict_New(); \
xddr = PyDict_New(); \
PyDict_SetItemString(dtop, xdn, xdd); \
PyDict_SetItemString(xd, xdn "_", xddr); \
PyDict_SetItemString(xd, xdn, xdd);

#define addConstInDict2(xd,xxd,xxdr,xs,xv) \
v= PyInt_FromLong((long)xv);\
s= PyString_FromString(xs);\
PyDict_SetItem(xd,s,s);\
PyDict_SetItem(xxd,s,v);\
PyDict_SetItem(xxdr,v,s);\
Py_DECREF(v);\
Py_DECREF(s); 

#define addStringInDict(xd,xs) \
s= PyString_FromString(xs);\
PyDict_SetItemString(xd,xs,s);\
Py_DECREF(s); 

#define addStringInDict2(xd,xxd,xs) \
s= PyString_FromString(xs);\
PyDict_SetItemString(xxd,xs,s);\
PyDict_SetItemString(xd,xs,s);\
Py_DECREF(s); 

void midleveldictionnary_init(PyObject *d)
{
  PyObject *v, *s, *dr, *ddr, *dtop;

  /* This top dictionnary stores the names of the enum dictionnaries */
  dtop = PyDict_New();
  s= PyString_FromString("Enumerates");

  /* d --> the CGNS module dictionnary */
  PyDict_SetItem(d, s, dtop);
  
  /* ----------- Open modes -1- */
  createDict(d, "OpenMode", dr, ddr);
  addConstInDict2(d,dr,ddr,"CG_MODE_READ"  ,CG_MODE_READ);
  addConstInDict2(d,dr,ddr,"CG_MODE_WRITE" ,CG_MODE_WRITE);
  addConstInDict2(d,dr,ddr,"CG_MODE_MODIFY",CG_MODE_MODIFY);
  addConstInDict2(d,dr,ddr,"CG_MODE_CLOSED",CG_MODE_CLOSED);
  addConstInDict2(d,dr,ddr,"MODE_READ"  ,   CG_MODE_READ);
  addConstInDict2(d,dr,ddr,"MODE_WRITE" ,   CG_MODE_WRITE);
#if CGNS_VERSION < 3000
  addConstInDict2(d,dr,ddr,"MODE_MODIFY",   2);
  addConstInDict2(d,dr,ddr,"MODE_CLOSED",   3);
#else
  addConstInDict2(d,dr,ddr,"MODE_MODIFY",   3);
  addConstInDict2(d,dr,ddr,"MODE_CLOSED",   2);
#endif
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  /* ----------- Error codes -2- */
  createDict(d, "ErrorCode", dr, ddr);
  addConstInDict2(d,dr,ddr,"CG_OK"  ,          CG_OK);
  addConstInDict2(d,dr,ddr,"CG_ERROR" ,        CG_ERROR);
  addConstInDict2(d,dr,ddr,"CG_NODE_NOT_FOUND",CG_NODE_NOT_FOUND);
  addConstInDict2(d,dr,ddr,"CG_INCORRECT_PATH",CG_INCORRECT_PATH);
  addConstInDict2(d,dr,ddr,"CG_NO_INDEX_DIM",  CG_NO_INDEX_DIM);
  addConstInDict2(d,dr,ddr,"ALL_OK"  ,         CG_OK);
  addConstInDict2(d,dr,ddr,"ERROR" ,           CG_ERROR);
  addConstInDict2(d,dr,ddr,"NODE_NOT_FOUND",   CG_NODE_NOT_FOUND);
  addConstInDict2(d,dr,ddr,"INCORRECT_PATH",   CG_INCORRECT_PATH);
  addConstInDict2(d,dr,ddr,"NO_INDEX_DIM",     CG_NO_INDEX_DIM);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- Error codes -2- */
  createDict(d, "Configuration", dr, ddr);
  addConstInDict2(d,dr,ddr,"CG_CONFIG_ERROR",   CG_CONFIG_ERROR);
  addConstInDict2(d,dr,ddr,"CG_CONFIG_COMPRESS",CG_CONFIG_COMPRESS);
  addConstInDict2(d,dr,ddr,"CG_CONFIG_SET_PATH",CG_CONFIG_SET_PATH);
  addConstInDict2(d,dr,ddr,"CG_CONFIG_ADD_PATH",CG_CONFIG_ADD_PATH);

  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- UNITS -3-4-5-6-7- */

  createDict(d,  "MassUnits", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",MassUnitsNull); /* not defined twice elsewhere */
  addConstInDict2(d,dr,ddr,"UserDefined",MassUnitsUserDefined); /* idem */
  addConstInDict2(d,dr,ddr,"Kilogram",Kilogram);
  addConstInDict2(d,dr,ddr,"Gram",Gram);
  addConstInDict2(d,dr,ddr,"Slug",Slug);
  addConstInDict2(d,dr,ddr,"PoundMass",PoundMass);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d,  "LengthUnits", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",LengthUnitsNull);
  addConstInDict2(d,dr,ddr,"UserDefined",LengthUnitsUserDefined);
  addConstInDict2(d,dr,ddr,"Meter",Meter);
  addConstInDict2(d,dr,ddr,"Centimeter",Centimeter);
  addConstInDict2(d,dr,ddr,"Millimeter",Millimeter);
  addConstInDict2(d,dr,ddr,"Foot",Foot);
  addConstInDict2(d,dr,ddr,"Inch",Inch);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "TimeUnits", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",TimeUnitsNull);
  addConstInDict2(d,dr,ddr,"UserDefined",TimeUnitsUserDefined);
  addConstInDict2(d,dr,ddr,"Second",Second);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "TemperatureUnits", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",TemperatureUnitsNull);
  addConstInDict2(d,dr,ddr,"UserDefined",TemperatureUnitsUserDefined);
  addConstInDict2(d,dr,ddr,"Kelvin",Kelvin);
  addConstInDict2(d,dr,ddr,"Celcius",Celcius);
  addConstInDict2(d,dr,ddr,"Rankine",Rankine);
  addConstInDict2(d,dr,ddr,"Fahrenheit",Fahrenheit);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "AngleUnits", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",AngleUnitsNull);
  addConstInDict2(d,dr,ddr,"UserDefined",AngleUnitsUserDefined);
  addConstInDict2(d,dr,ddr,"Degree",Degree);
  addConstInDict2(d,dr,ddr,"Radian",Radian);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  /* ----------- DataClass_t -8- */
  createDict(d, "DataClass", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",DataClassNull);
  addConstInDict2(d,dr,ddr,"UserDefined",DataClassUserDefined);
  addConstInDict2(d,dr,ddr,"Dimensional",Dimensional);
  addConstInDict2(d,dr,ddr,"NormalizedByDimensional",NormalizedByDimensional);
  addConstInDict2(d,dr,ddr,"NormalizedByUnknownDimensional",NormalizedByUnknownDimensional);
  addConstInDict2(d,dr,ddr,"NondimensionalParameter",NondimensionalParameter);
  addConstInDict2(d,dr,ddr,"DimensionlessConstant",DimensionlessConstant);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- GridLocation_t -9- */
  createDict(d, "GridLocation", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",GridLocationNull);
  addConstInDict2(d,dr,ddr,"UserDefined",GridLocationUserDefined);
  addConstInDict2(d,dr,ddr,"Vertex",Vertex);
  addConstInDict2(d,dr,ddr,"CellCenter",CellCenter);
  addConstInDict2(d,dr,ddr,"FaceCenter",FaceCenter);
  addConstInDict2(d,dr,ddr,"IFaceCenter",IFaceCenter);
  addConstInDict2(d,dr,ddr,"JFaceCenter",JFaceCenter);
  addConstInDict2(d,dr,ddr,"KFaceCenter",KFaceCenter);
  addConstInDict2(d,dr,ddr,"EdgeCenter",EdgeCenter);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- BCDataType_t -10- */
  createDict(d, "BCDataType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",BCDataTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",BCDataTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Dirichlet",Dirichlet);
  addConstInDict2(d,dr,ddr,"Neumann",Neumann);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- GridConnectivityType_t -11- */
  createDict(d, "GridConnectivityType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",GridConnectivityTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",GridConnectivityTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Overset",Overset);
  addConstInDict2(d,dr,ddr,"Abutting",Abutting);
  addConstInDict2(d,dr,ddr,"Abutting1to1",Abutting1to1);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- PointSetType_t -12- */
  createDict(d, "PointSetType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",PointSetTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",PointSetTypeUserDefined);
  addConstInDict2(d,dr,ddr,"PointList",PointList);
  addConstInDict2(d,dr,ddr,"PointListDonor",PointListDonor);
  addConstInDict2(d,dr,ddr,"PointRange",PointRange);
  addConstInDict2(d,dr,ddr,"PointRangeDonor",PointRangeDonor);
  addConstInDict2(d,dr,ddr,"ElementRange",ElementRange);
  addConstInDict2(d,dr,ddr,"ElementList",ElementList);
  addConstInDict2(d,dr,ddr,"CellListDonor",CellListDonor);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- GoverningEquationsType_t -13- */
  createDict(d, "GoverningEquationsType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",GoverningEquationsNull); 
  addConstInDict2(d,dr,ddr,"UserDefined",GoverningEquationsUserDefined);
  addConstInDict2(d,dr,ddr,"FullPotential",FullPotential);
  addConstInDict2(d,dr,ddr,"Euler",Euler);
  addConstInDict2(d,dr,ddr,"NSLaminar",NSLaminar);
  addConstInDict2(d,dr,ddr,"NSTurbulent",NSTurbulent);
  addConstInDict2(d,dr,ddr,"NSLaminarIncompressible",NSLaminarIncompressible);
  addConstInDict2(d,dr,ddr,"NSTurbulentIncompressible",NSTurbulentIncompressible);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- ModelType_t -14- (and associated) */
  createDict(d, "ModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Ideal",Ideal);
  addConstInDict2(d,dr,ddr,"VanderWaals",VanderWaals);
  addConstInDict2(d,dr,ddr,"Constant",Constant);
  addConstInDict2(d,dr,ddr,"PowerLaw",PowerLaw);
  addConstInDict2(d,dr,ddr,"SutherlandLaw",SutherlandLaw);
  addConstInDict2(d,dr,ddr,"ConstantPrandtl",ConstantPrandtl);
  addConstInDict2(d,dr,ddr,"EddyViscosity",EddyViscosity);
  addConstInDict2(d,dr,ddr,"ReynoldsStress",ReynoldsStress);
  addConstInDict2(d,dr,ddr,"ReynoldsStressAlgebraic",ReynoldsStressAlgebraic);
  addConstInDict2(d,dr,ddr,"Algebraic_BaldwinLomax",Algebraic_BaldwinLomax);
  addConstInDict2(d,dr,ddr,"Algebraic_CebeciSmith",Algebraic_CebeciSmith);
  addConstInDict2(d,dr,ddr,"HalfEquation_JohnsonKing",HalfEquation_JohnsonKing);
  addConstInDict2(d,dr,ddr,"OneEquation_BaldwinBarth",OneEquation_BaldwinBarth);
  addConstInDict2(d,dr,ddr,"OneEquation_SpalartAllmaras",OneEquation_SpalartAllmaras);
  addConstInDict2(d,dr,ddr,"TwoEquation_JonesLaunder",TwoEquation_JonesLaunder);
  addConstInDict2(d,dr,ddr,"TwoEquation_MenterSST",TwoEquation_MenterSST);
  addConstInDict2(d,dr,ddr,"TwoEquation_Wilcox",TwoEquation_Wilcox);
  addConstInDict2(d,dr,ddr,"CaloricallyPerfect",CaloricallyPerfect);
  addConstInDict2(d,dr,ddr,"ThermallyPerfect",ThermallyPerfect);
  addConstInDict2(d,dr,ddr,"ConstantDensity",ConstantDensity);
  addConstInDict2(d,dr,ddr,"RedlichKwong",RedlichKwong);
  addConstInDict2(d,dr,ddr,"Frozen",Frozen);
  addConstInDict2(d,dr,ddr,"ThermalEquilib",ThermalEquilib);
  addConstInDict2(d,dr,ddr,"ThermalNonequilib",ThermalNonequilib);
  addConstInDict2(d,dr,ddr,"ChemicalEquilibCurveFit",ChemicalEquilibCurveFit);
  addConstInDict2(d,dr,ddr,"ChemicalEquilibMinimization",ChemicalEquilibMinimization);
  addConstInDict2(d,dr,ddr,"ChemicalNonequilib",ChemicalNonequilib);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "GasModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Ideal",Ideal);
  addConstInDict2(d,dr,ddr,"VanderWaals",VanderWaals);
  addConstInDict2(d,dr,ddr,"RedlichKwong",RedlichKwong);
  addConstInDict2(d,dr,ddr,"CaloricallyPerfect",CaloricallyPerfect);
  addConstInDict2(d,dr,ddr,"ThermallyPerfect",ThermallyPerfect);
  addConstInDict2(d,dr,ddr,"ConstantDensity",ConstantDensity);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "ViscosityModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Constant",Constant);
  addConstInDict2(d,dr,ddr,"PowerLaw",PowerLaw);
  addConstInDict2(d,dr,ddr,"SutherlandLaw",SutherlandLaw);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  createDict(d, "ThermalConductivityModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Constant",Constant);
  addConstInDict2(d,dr,ddr,"PowerLaw",PowerLaw);
  addConstInDict2(d,dr,ddr,"SutherlandLaw",SutherlandLaw);
  addConstInDict2(d,dr,ddr,"ConstantPrandtl",ConstantPrandtl);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  createDict(d, "TurbulenceClosureType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"EddyViscosity",EddyViscosity);
  addConstInDict2(d,dr,ddr,"ReynoldsStress",ReynoldsStress);
  addConstInDict2(d,dr,ddr,"ReynoldsStressAlgebraic",ReynoldsStressAlgebraic);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  createDict(d, "TurbulenceModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Algebraic_BaldwinLomax",Algebraic_BaldwinLomax);
  addConstInDict2(d,dr,ddr,"Algebraic_CebeciSmith",Algebraic_CebeciSmith);
  addConstInDict2(d,dr,ddr,"HalfEquation_JohnsonKing",HalfEquation_JohnsonKing);
  addConstInDict2(d,dr,ddr,"OneEquation_BaldwinBarth",OneEquation_BaldwinBarth);
  addConstInDict2(d,dr,ddr,"OneEquation_SpalartAllmaras",OneEquation_SpalartAllmaras);
  addConstInDict2(d,dr,ddr,"TwoEquation_JonesLaunder",TwoEquation_JonesLaunder);
  addConstInDict2(d,dr,ddr,"TwoEquation_MenterSST",TwoEquation_MenterSST);
  addConstInDict2(d,dr,ddr,"TwoEquation_Wilcox",TwoEquation_Wilcox);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  createDict(d, "ThermalRelaxationModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Frozen",Frozen);
  addConstInDict2(d,dr,ddr,"ThermalEquilib",ThermalEquilib);
  addConstInDict2(d,dr,ddr,"ThermalNonequilib",ThermalNonequilib);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  createDict(d, "ChemicalKineticsModelType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ModelTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ModelTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Frozen",Frozen);
  addConstInDict2(d,dr,ddr,"ChemicalEquilibCurveFit",ChemicalEquilibCurveFit);
  addConstInDict2(d,dr,ddr,"ChemicalEquilibMinimization",ChemicalEquilibMinimization);
  addConstInDict2(d,dr,ddr,"ChemicalNonequilib",ChemicalNonequilib);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- BCType_t -15- */
  createDict(d, "BCType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",BCTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",BCTypeUserDefined);
  addConstInDict2(d,dr,ddr,"BCAxisymmetricWedge",BCAxisymmetricWedge);
  addConstInDict2(d,dr,ddr,"BCDegenerateLine",BCDegenerateLine);
  addConstInDict2(d,dr,ddr,"BCDegeneratePoint",BCDegeneratePoint);
  addConstInDict2(d,dr,ddr,"BCDirichlet",BCDirichlet);
  addConstInDict2(d,dr,ddr,"BCExtrapolate",BCExtrapolate);
  addConstInDict2(d,dr,ddr,"BCFarfield",BCFarfield);
  addConstInDict2(d,dr,ddr,"BCGeneral",BCGeneral);
  addConstInDict2(d,dr,ddr,"BCInflow",BCInflow);
  addConstInDict2(d,dr,ddr,"BCInflowSubsonic",BCInflowSubsonic);
  addConstInDict2(d,dr,ddr,"BCInflowSupersonic",BCInflowSupersonic);
  addConstInDict2(d,dr,ddr,"BCNeumann",BCNeumann);
  addConstInDict2(d,dr,ddr,"BCOutflow",BCOutflow);
  addConstInDict2(d,dr,ddr,"BCOutflowSubsonic",BCOutflowSubsonic);
  addConstInDict2(d,dr,ddr,"BCOutflowSupersonic",BCOutflowSupersonic);
  addConstInDict2(d,dr,ddr,"BCSymmetryPlane",BCSymmetryPlane);
  addConstInDict2(d,dr,ddr,"BCSymmetryPolar",BCSymmetryPolar);
  addConstInDict2(d,dr,ddr,"BCTunnelInflow",BCTunnelInflow);
  addConstInDict2(d,dr,ddr,"BCTunnelOutflow",BCTunnelOutflow);
  addConstInDict2(d,dr,ddr,"BCWall",BCWall);
  addConstInDict2(d,dr,ddr,"BCWallInviscid",BCWallInviscid);
  addConstInDict2(d,dr,ddr,"BCWallViscous",BCWallViscous);
  addConstInDict2(d,dr,ddr,"BCWallViscousHeatFlux",BCWallViscousHeatFlux);
  addConstInDict2(d,dr,ddr,"BCWallViscousIsothermal",BCWallViscousIsothermal);
  addConstInDict2(d,dr,ddr,"FamilySpecified",FamilySpecified);
  Py_DECREF(dr);
  Py_DECREF(ddr);
  
  /* ----------- DataType_t -16- */
  createDict(d, "DataType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",DataTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",DataTypeUserDefined);
  addConstInDict2(d,dr,ddr,Integer_s,Integer);
  addConstInDict2(d,dr,ddr,RealSingle_s,RealSingle);
  addConstInDict2(d,dr,ddr,RealDouble_s,RealDouble);
  addConstInDict2(d,dr,ddr,Character_s,Character);
  addConstInDict2(d,dr,ddr,"c",Character); /* Unknown collision (Numeric ?) */
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- ElementType_t -17- */
  createDict(d, "ElementType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ElementTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ElementTypeUserDefined);
  addConstInDict2(d,dr,ddr,"NODE",NODE);
  addConstInDict2(d,dr,ddr,"BAR_2",BAR_2);
  addConstInDict2(d,dr,ddr,"BAR_3",BAR_3);
  addConstInDict2(d,dr,ddr,"TRI_3",TRI_3);
  addConstInDict2(d,dr,ddr,"TRI_6",TRI_6);
  addConstInDict2(d,dr,ddr,"QUAD_4",QUAD_4);
  addConstInDict2(d,dr,ddr,"QUAD_8",QUAD_8);
  addConstInDict2(d,dr,ddr,"QUAD_9",QUAD_9);
  addConstInDict2(d,dr,ddr,"TETRA_4",TETRA_4);
  addConstInDict2(d,dr,ddr,"TETRA_10",TETRA_10);
  addConstInDict2(d,dr,ddr,"PYRA_5",PYRA_5);
  addConstInDict2(d,dr,ddr,"PYRA_14",PYRA_14);
  addConstInDict2(d,dr,ddr,"PENTA_6",PENTA_6);
  addConstInDict2(d,dr,ddr,"PENTA_15",PENTA_15);
  addConstInDict2(d,dr,ddr,"PENTA_18",PENTA_18);
  addConstInDict2(d,dr,ddr,"HEXA_8",HEXA_8);
  addConstInDict2(d,dr,ddr,"HEXA_20",HEXA_20);
  addConstInDict2(d,dr,ddr,"HEXA_27",HEXA_27);
  addConstInDict2(d,dr,ddr,"MIXED",MIXED);
  addConstInDict2(d,dr,ddr,"NGON_n",NGON_n);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- ZoneType_t -18- */
  createDict(d, "ZoneType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ZoneTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ZoneTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Structured",Structured);
  addConstInDict2(d,dr,ddr,"Unstructured",Unstructured);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- RigidGridMotionType_t -19- */
  createDict(d, "RigidGridMotionType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",RigidGridMotionTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",RigidGridMotionTypeUserDefined);
  addConstInDict2(d,dr,ddr,"ConstantRate",ConstantRate);
  addConstInDict2(d,dr,ddr,"VariableRate",VariableRate);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- ArbitraryGridMotionType_t -20- */
  createDict(d, "ArbitraryGridMotionType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",ArbitraryGridMotionTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",ArbitraryGridMotionTypeUserDefined);
  addConstInDict2(d,dr,ddr,"NonDeformingGrid",NonDeformingGrid);
  addConstInDict2(d,dr,ddr,"DeformingGrid",DeformingGrid);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- SimulationType_t -21- */
  createDict(d, "SimulationType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",SimulationTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",SimulationTypeUserDefined);
  addConstInDict2(d,dr,ddr,"TimeAccurate",TimeAccurate);
  addConstInDict2(d,dr,ddr,"NonTimeAccurate",NonTimeAccurate);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- WallFunctionType_t -22- */
  createDict(d, "WallFunctionType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",WallFunctionTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",WallFunctionTypeUserDefined);
  addConstInDict2(d,dr,ddr,"Generic",Generic);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- AreaType_t -23- */
  createDict(d, "AreaType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",AreaTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",AreaTypeUserDefined);
  addConstInDict2(d,dr,ddr,"BleedArea",BleedArea);
  addConstInDict2(d,dr,ddr,"CaptureArea",CaptureArea);
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- AverageInterfaceType_t -24- */
  createDict(d, "AverageInterfaceType", dr, ddr);
  addConstInDict2(d,dr,ddr,"Null",AverageInterfaceTypeNull);
  addConstInDict2(d,dr,ddr,"UserDefined",AverageInterfaceTypeUserDefined);
  addConstInDict2(d,dr,ddr,"AverageAll",AverageAll);
  addConstInDict2(d,dr,ddr,"AverageCircumferential",AverageCircumferential);
  addConstInDict2(d,dr,ddr,"AverageRadial",AverageRadial);
  addConstInDict2(d,dr,ddr,"AverageI",AverageI);
  addConstInDict2(d,dr,ddr,"AverageJ",AverageJ);
  addConstInDict2(d,dr,ddr,"AverageK",AverageK);  
  Py_DECREF(dr);
  Py_DECREF(ddr);

  /* ----------- Label strings */
  dr = PyDict_New();
  PyDict_SetItemString(d, "LabelString", dr);
  addStringInDict2(d,dr,"ArbitraryGridMotion_t")
  addStringInDict2(d,dr,"Area_t")
  addStringInDict2(d,dr,"AverageInterface_t")
  addStringInDict2(d,dr,"Axisymmetry_t")
  addStringInDict2(d,dr,"BCDataSet_t")
  addStringInDict2(d,dr,"BCData_t")
  addStringInDict2(d,dr,"BCProperty_t")
  addStringInDict2(d,dr,"BC_t")
  addStringInDict2(d,dr,"BaseIterativeData_t")
  addStringInDict2(d,dr,"CGNSBase_t")
  addStringInDict2(d,dr,"CGNSLibraryVersion_t")
  addStringInDict2(d,dr,"ChemicalKineticsModel_t")
  addStringInDict2(d,dr,"ConvergenceHistory_t")
  addStringInDict2(d,dr,"DataArray_t")
  addStringInDict2(d,dr,"DataClass_t")
  addStringInDict2(d,dr,"DataConversion_t")
  addStringInDict2(d,dr,"Descriptor_t")
  addStringInDict2(d,dr,"DimensionalExponents_t")
  addStringInDict2(d,dr,"DimensionalUnits_t")    
  addStringInDict2(d,dr,"DiscreteData_t")
  addStringInDict2(d,dr,"Elements_t")
  addStringInDict2(d,dr,"FamilyBC_t")
  addStringInDict2(d,dr,"FamilyName_t")
  addStringInDict2(d,dr,"Family_t")
  addStringInDict2(d,dr,"FlowEquationSet_t")
  addStringInDict2(d,dr,"FlowSolution_t")
  addStringInDict2(d,dr,"GasModel_t")
  addStringInDict2(d,dr,"GeometryEntity_t")
  addStringInDict2(d,dr,"GeometryFile_t")
  addStringInDict2(d,dr,"GeometryFormat_t")
  addStringInDict2(d,dr,"GeometryReference_t")
  addStringInDict2(d,dr,"GoverningEquations_t")
  addStringInDict2(d,dr,"Gravity_t")
  addStringInDict2(d,dr,"GridConnectivity1to1_t")
  addStringInDict2(d,dr,"GridConnectivityProperty_t")
  addStringInDict2(d,dr,"GridConnectivityType_t")
  addStringInDict2(d,dr,"GridConnectivity_t")
  addStringInDict2(d,dr,"GridCoordinates_t")
  addStringInDict2(d,dr,"GridLocation_t")
  addStringInDict2(d,dr,"IndexArray_t")
  addStringInDict2(d,dr,"IndexRange_t")    
  addStringInDict2(d,dr,"IntegralData_t")
  addStringInDict2(d,dr,"InwardNormalList_t")
  addStringInDict2(d,dr,"Ordinal_t")
  addStringInDict2(d,dr,"OversetHoles_t")
  addStringInDict2(d,dr,"Periodic_t")
  addStringInDict2(d,dr,"ReferenceState_t")
  addStringInDict2(d,dr,"RigidGridMotion_t")
  addStringInDict2(d,dr,"Rind_t")    
  addStringInDict2(d,dr,"RotatingCoordinates_t")
  addStringInDict2(d,dr,"SimulationType_t")
  addStringInDict2(d,dr,"ThermalConductivityModel_t")
  addStringInDict2(d,dr,"ThermalRelaxationModel_t")
  addStringInDict2(d,dr,"TurbulenceClosure_t")
  addStringInDict2(d,dr,"TurbulenceModel_t")
  addStringInDict2(d,dr,"UserDefinedData_t")
  addStringInDict2(d,dr,"ViscosityModel_t")
  addStringInDict2(d,dr,"WallFunction_t")
  addStringInDict2(d,dr,"ZoneBC_t")
  addStringInDict2(d,dr,"ZoneGridConnectivity_t")
  addStringInDict2(d,dr,"ZoneIterativeData_t")
  addStringInDict2(d,dr,"ZoneType_t")
  addStringInDict2(d,dr,"Zone_t")
  Py_DECREF(dr);
  
  /* ----------- NAMES - All these are stored as string valued variables */
  dr = PyDict_New();
  PyDict_SetItemString(d, "Names", dr);
  addStringInDict2(d,dr,Acoustic_s);
  addStringInDict2(d,dr,ArbitraryGridMotionPointers_s);
  addStringInDict2(d,dr,CharacteristicAcousticMinus_s);
  addStringInDict2(d,dr,CharacteristicAcousticPlus_s);
  addStringInDict2(d,dr,CharacteristicEntropy_s);
  addStringInDict2(d,dr,CharacteristicVorticity1_s);
  addStringInDict2(d,dr,CharacteristicVorticity2_s);
  addStringInDict2(d,dr,CoefDrag_s);
  addStringInDict2(d,dr,CoefLift_s);
  addStringInDict2(d,dr,CoefMomentEta_s);
  addStringInDict2(d,dr,CoefMomentPhi_s);
  addStringInDict2(d,dr,CoefMomentR_s);
  addStringInDict2(d,dr,CoefMomentTheta_s);
  addStringInDict2(d,dr,CoefMomentX_s);
  addStringInDict2(d,dr,CoefMomentXi_s);
  addStringInDict2(d,dr,CoefMomentY_s);
  addStringInDict2(d,dr,CoefMomentZ_s);
  addStringInDict2(d,dr,CoefMomentZeta_s);
  addStringInDict2(d,dr,CoefPressure_s);
  addStringInDict2(d,dr,CoefSkinFrictionX_s);
  addStringInDict2(d,dr,CoefSkinFrictionY_s);
  addStringInDict2(d,dr,CoefSkinFrictionZ_s);
  addStringInDict2(d,dr,Coef_Area_s);
  addStringInDict2(d,dr,Coef_Length_s);
  addStringInDict2(d,dr,Coef_PressureDynamic_s);
  addStringInDict2(d,dr,Coef_PressureDynamic_s);
  addStringInDict2(d,dr,Coef_PressureReference_s);
  addStringInDict2(d,dr,CoordinateEta_s);
  addStringInDict2(d,dr,CoordinateNormal_s);
  addStringInDict2(d,dr,CoordinatePhi_s);
  addStringInDict2(d,dr,CoordinateR_s);
  addStringInDict2(d,dr,CoordinateTangential_s);
  addStringInDict2(d,dr,CoordinateTheta_s);
  addStringInDict2(d,dr,CoordinateTransform_s);
  addStringInDict2(d,dr,CoordinateX_s);
  addStringInDict2(d,dr,CoordinateX_s);
  addStringInDict2(d,dr,CoordinateXi_s);
  addStringInDict2(d,dr,CoordinateY_s);
  addStringInDict2(d,dr,CoordinateZ_s);
  addStringInDict2(d,dr,CoordinateZeta_s);
  addStringInDict2(d,dr,DensityStagnation_s);
  addStringInDict2(d,dr,Density_s);
  addStringInDict2(d,dr,Drag_s);
  addStringInDict2(d,dr,ElementConnectivity_s);
  addStringInDict2(d,dr,EnergyInternal_s);
  addStringInDict2(d,dr,EnergyKinetic_s);
  addStringInDict2(d,dr,EnergyStagnationDensity_s);
  addStringInDict2(d,dr,EnergyStagnation_s);
  addStringInDict2(d,dr,EnthalpyStagnation_s);
  addStringInDict2(d,dr,Enthalpy_s);
  addStringInDict2(d,dr,EntropyApprox_s);
  addStringInDict2(d,dr,Entropy_s);
  addStringInDict2(d,dr,FamilyPointers_s);
  addStringInDict2(d,dr,FlowSolutionPointers_s);
  addStringInDict2(d,dr,ForcePhi_s);
  addStringInDict2(d,dr,ForceR_s);
  addStringInDict2(d,dr,ForceTheta_s);
  addStringInDict2(d,dr,ForceX_s);
  addStringInDict2(d,dr,ForceY_s);
  addStringInDict2(d,dr,ForceZ_s);
  addStringInDict2(d,dr,FuelAirRatio_s);
  addStringInDict2(d,dr,GridCoordinatesPointers_s);
  addStringInDict2(d,dr,GridVelocityEta_s);
  addStringInDict2(d,dr,GridVelocityPhi_s);
  addStringInDict2(d,dr,GridVelocityR_s);
  addStringInDict2(d,dr,GridVelocityTheta_s);
  addStringInDict2(d,dr,GridVelocityX_s);
  addStringInDict2(d,dr,GridVelocityXi_s);
  addStringInDict2(d,dr,GridVelocityY_s);
  addStringInDict2(d,dr,GridVelocityZ_s);
  addStringInDict2(d,dr,GridVelocityZeta_s);
  addStringInDict2(d,dr,HeatOfFormation_ps);
  addStringInDict2(d,dr,HeatOfFormation_s);
  addStringInDict2(d,dr,IdealGasConstant_s);
  addStringInDict2(d,dr,InterpolantsDonor_s);
  addStringInDict2(d,dr,IterationValues_s);
  addStringInDict2(d,dr,LaminarViscosity_ps);
  addStringInDict2(d,dr,LaminarViscosity_s);
  addStringInDict2(d,dr,LengthReference_s);
  addStringInDict2(d,dr,Lift_s);
  addStringInDict2(d,dr,Mach_VelocitySound_s);
  addStringInDict2(d,dr,Mach_Velocity_s);
  addStringInDict2(d,dr,Mach_s);
  addStringInDict2(d,dr,MassFlow_s);
  addStringInDict2(d,dr,MassFraction_ps);
  addStringInDict2(d,dr,MassFraction_s);
  addStringInDict2(d,dr,MoleFraction_ps);
  addStringInDict2(d,dr,MoleFraction_s);
  addStringInDict2(d,dr,MolecularWeight_ps);
  addStringInDict2(d,dr,MolecularWeight_s);
  addStringInDict2(d,dr,MomentEta_s);
  addStringInDict2(d,dr,MomentPhi_s);
  addStringInDict2(d,dr,MomentR_s);
  addStringInDict2(d,dr,MomentTheta_s);
  addStringInDict2(d,dr,MomentX_s);
  addStringInDict2(d,dr,MomentXi_s);
  addStringInDict2(d,dr,MomentY_s);
  addStringInDict2(d,dr,MomentZ_s);
  addStringInDict2(d,dr,MomentZeta_s);
  addStringInDict2(d,dr,Moment_CenterX_s);
  addStringInDict2(d,dr,Moment_CenterY_s);
  addStringInDict2(d,dr,Moment_CenterZ_s);
  addStringInDict2(d,dr,MomentumMagnitude_s);
  addStringInDict2(d,dr,MomentumX_s);
  addStringInDict2(d,dr,MomentumY_s);
  addStringInDict2(d,dr,MomentumZ_s);
  addStringInDict2(d,dr,NumberOfFamilies_s);
  addStringInDict2(d,dr,NumberOfZones_s);
  addStringInDict2(d,dr,OriginLocation_s);
  addStringInDict2(d,dr,ParentData_s);
  addStringInDict2(d,dr,Potential_s);
  addStringInDict2(d,dr,PowerLawExponent_s);
  addStringInDict2(d,dr,PrandtlTurbulent_s);
  addStringInDict2(d,dr,Prandtl_SpecificHeatPressure_s);
  addStringInDict2(d,dr,Prandtl_ThermalConductivity_s);
  addStringInDict2(d,dr,Prandtl_ViscosityMolecular_s);
  addStringInDict2(d,dr,Prandtl_s);
  addStringInDict2(d,dr,PressureDynamic_s);
  addStringInDict2(d,dr,PressureStagnation_s);
  addStringInDict2(d,dr,Pressure_s);
  addStringInDict2(d,dr,ReynoldsStressXX_s);
  addStringInDict2(d,dr,ReynoldsStressXY_s);
  addStringInDict2(d,dr,ReynoldsStressXZ_s);
  addStringInDict2(d,dr,ReynoldsStressYY_s);
  addStringInDict2(d,dr,ReynoldsStressYZ_s);
  addStringInDict2(d,dr,ReynoldsStressZZ_s);
  addStringInDict2(d,dr,Reynolds_Length_s);
  addStringInDict2(d,dr,Reynolds_Velocity_s);
  addStringInDict2(d,dr,Reynolds_ViscosityKinematic_s);
  addStringInDict2(d,dr,Reynolds_s);
  addStringInDict2(d,dr,RiemannInvariantMinus_s);
  addStringInDict2(d,dr,RiemannInvariantPlus_s);
  addStringInDict2(d,dr,RigidGridMotionPointers_s);
  addStringInDict2(d,dr,RigidRotationAngle_s);
  addStringInDict2(d,dr,RigidRotationRate_s);
  addStringInDict2(d,dr,RigidVelocity_s);
  addStringInDict2(d,dr,SkinFrictionMagnitude_s);
  addStringInDict2(d,dr,SkinFrictionX_s);
  addStringInDict2(d,dr,SkinFrictionY_s);
  addStringInDict2(d,dr,SkinFrictionZ_s);
  addStringInDict2(d,dr,SpecificHeatPressure_s);
  addStringInDict2(d,dr,SpecificHeatRatio_Pressure_s);
  addStringInDict2(d,dr,SpecificHeatRatio_Volume_s);
  addStringInDict2(d,dr,SpecificHeatRatio_s);
  addStringInDict2(d,dr,SpecificHeatVolume_s);
  addStringInDict2(d,dr,StreamFunction_s);
  addStringInDict2(d,dr,SutherlandLawConstant_s);
  addStringInDict2(d,dr,TemperatureReference_s);
  addStringInDict2(d,dr,TemperatureStagnation_s);
  addStringInDict2(d,dr,Temperature_s);
  addStringInDict2(d,dr,ThermalConductivityReference_s);
  addStringInDict2(d,dr,ThermalConductivity_ps);
  addStringInDict2(d,dr,ThermalConductivity_s);
  addStringInDict2(d,dr,TimeValues_s);
  addStringInDict2(d,dr,TurbulentBBReynolds_s);
  addStringInDict2(d,dr,TurbulentDissipationRate_s);
  addStringInDict2(d,dr,TurbulentDissipation_s);
  addStringInDict2(d,dr,TurbulentDistance_s);
  addStringInDict2(d,dr,TurbulentEnergyKinetic_s);
  addStringInDict2(d,dr,TurbulentSANuTilde_s);
  addStringInDict2(d,dr,VectorMagnitude_ps);
  addStringInDict2(d,dr,VectorNormal_ps);
  addStringInDict2(d,dr,VectorPhi_ps);
  addStringInDict2(d,dr,VectorTangential_ps);
  addStringInDict2(d,dr,VectorTheta_ps);
  addStringInDict2(d,dr,VectorX_ps);
  addStringInDict2(d,dr,VectorY_ps);
  addStringInDict2(d,dr,VectorZ_ps);
  addStringInDict2(d,dr,VelocityAngleX_s);
  addStringInDict2(d,dr,VelocityAngleY_s);
  addStringInDict2(d,dr,VelocityAngleZ_s);
  addStringInDict2(d,dr,VelocityMagnitude_s);
  addStringInDict2(d,dr,VelocityNormal_s);
  addStringInDict2(d,dr,VelocityPhi_s);
  addStringInDict2(d,dr,VelocityR_s);
  addStringInDict2(d,dr,VelocitySoundStagnation_s);
  addStringInDict2(d,dr,VelocitySound_s);
  addStringInDict2(d,dr,VelocityTangential_s);
  addStringInDict2(d,dr,VelocityTheta_s);
  addStringInDict2(d,dr,VelocityUnitVectorX_s);
  addStringInDict2(d,dr,VelocityUnitVectorY_s);
  addStringInDict2(d,dr,VelocityUnitVectorZ_s);
  addStringInDict2(d,dr,VelocityX_s);
  addStringInDict2(d,dr,VelocityY_s);
  addStringInDict2(d,dr,VelocityZ_s);
  addStringInDict2(d,dr,VibrationalElectronEnergy_s);
  addStringInDict2(d,dr,VibrationalElectronTemperature_s);
  addStringInDict2(d,dr,ViscosityEddyDynamic_s);
  addStringInDict2(d,dr,ViscosityEddy_s);
  addStringInDict2(d,dr,ViscosityKinematic_s);
  addStringInDict2(d,dr,ViscosityMolecularReference_s);
  addStringInDict2(d,dr,ViscosityMolecular_s);
  addStringInDict2(d,dr,VorticityMagnitude_s);
  addStringInDict2(d,dr,VorticityX_s);
  addStringInDict2(d,dr,VorticityY_s);
  addStringInDict2(d,dr,VorticityZ_s);
  addStringInDict2(d,dr,ZonePointers_s);
  Py_DECREF(dr);
  
  Py_DECREF(dtop);
}
