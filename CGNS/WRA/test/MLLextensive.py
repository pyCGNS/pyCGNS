#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
# "Testing reveals the presence of bugs, not their absence" D.Knuth
#
# THIS IS A SAMPLE STRUCTURE - NO MEANINGFULL VALUES FOR COMPUTATION
# THIS TEST CALLS AT LEAST ONCE EVERY MLL WRAPPED FUNCTION
# (about 250 calls to MLL)
#
from __future__ import print_function
import time
from CGNS import pyCGNSconfig

try:
    if not pyCGNSconfig.HAS_MLL:
        import sys

        print("### pyCGNS: NOT built with MLL/Python binding")
        print("### pyCGNS: Cannot perform MLL tests")
        sys.exit(1)
except KeyError:
    print("### pyCGNS: Cannot find pyCGNS config ?")
    sys.exit(1)

import CGNS
import CGNS.WRA.wrapper as Mll
import numpy       as N
import CGNS.WRA.utilities  as U
import CGNS.PAT.cgnskeywords as CGK

import os
import posixpath

TDBNAME1 = "mll-1.cgns"
TDBNAME2 = "mll-2.cgns"
TDBNAME3 = "mll-3.cgns"
TDBNAME4 = "mll-4.cgns"


class xFAILED(Exception):
    def __init__(self):
        print("#### *** FAILED ON LAST TEST ***")


# ----------------------------------------------------------------------
def tt(s, n=TDBNAME1):
    import posixpath
    import os
    if (n != '') and posixpath.exists(n):
        os.unlink(n)
    print("#---#")
    print(s)


# ----------------------------------------------------------------------
def ee(a, code=0):
    if a.error[0] != 0:
        if (code == 0) or (a.error[0] != code):
            print("#   # Error %d [%s]" % (a.error[0], a.error[1]))
            raise xFAILED
        else:
            print("#   # Got expected error [%s]" % a.error[1])


# ----------------------------------------------------------------------
_time_start = 0


def start():
    global _time_start
    _time_start = time.time()


def stop():
    _time_stop = time.time()
    print("#   # time %.2fs" % (_time_stop - _time_start))


# ----------------------------------------------------------------------
def acube(im=3, jm=5, km=7, offset=0):
    # inverse k/i in order to get correct order in ADF file
    x = N.zeros((km, jm, im), 'd')
    y = N.zeros((km, jm, im), 'd')
    z = N.zeros((km, jm, im), 'd')
    for i in range(im):
        for j in range(jm):
            for k in range(km):
                x[k, j, i] = i + (im - 1) * offset
                y[k, j, i] = j
                z[k, j, i] = k
    return (x, y, z)


c01 = acube()
c02 = acube(offset=1)
c03 = acube(offset=2)
d = {}
d['Density'] = acube(2, 4, 6)[0] * 1.02
d['Pressure'] = acube(2, 4, 6)[1] * 1.08
d['Temperature'] = acube(2, 4, 6)[2] * 1.14

# -------------------------------------------------------------------------
# test for MLL wrapper
print("-" * 70)
print("#   # MLL Trivial calls to every API function")

# -------------------------------------------------------------------------
tt("# 01# Open a database", TDBNAME2)
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_WRITE)
ee(a)

tt("# 02# Base", '')
a.basewrite('Base', 3, 3)
ee(a)

tt("# 03# Zones", '')
a.zonewrite(1, 'Zone 01', [3, 5, 7, 2, 4, 6, 0, 0, 0], CGK.Structured)
ee(a)
a.zonewrite(1, 'Zone 02', [3, 5, 7, 2, 4, 6, 0, 0, 0], CGK.Structured)
ee(a)
a.zonewrite(1, 'Zone 03', [3, 5, 7, 2, 4, 6, 0, 0, 0], CGK.Structured)
ee(a)

tt("# 04# Coordinates", '')
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateX, c01[0])
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateY, c01[1])
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateZ, c01[2])
ee(a)
a.coordwrite(1, 2, CGK.RealDouble, CGK.CoordinateX, c02[0])
ee(a)
a.coordwrite(1, 2, CGK.RealDouble, CGK.CoordinateY, c02[1])
ee(a)
a.coordwrite(1, 2, CGK.RealDouble, CGK.CoordinateZ, c02[2])
ee(a)
a.coordwrite(1, 3, CGK.RealDouble, CGK.CoordinateX, c03[0])
ee(a)
a.coordwrite(1, 3, CGK.RealDouble, CGK.CoordinateY, c03[1])
ee(a)
a.coordwrite(1, 3, CGK.RealDouble, CGK.CoordinateZ, c03[2])
ee(a)

tt("# 05# Connectivity", '')
a.one2onewrite(1, 1, "[01-02]", "Zone 02", (3, 1, 1, 3, 5, 7), (1, 1, 1, 1, 5, 7), (1, 2, 3))
ee(a)
a.one2onewrite(1, 2, "[02-03]", "Zone 03", (3, 1, 1, 3, 5, 7), (1, 1, 1, 1, 5, 7), (1, 2, 3))
ee(a)
a.one2onewrite(1, 2, "[02-01]", "Zone 01", (3, 5, 7, 3, 1, 1), (1, 5, 7, 1, 1, 1), (1, 2, 3))
ee(a)
a.one2onewrite(1, 3, "[03-02]", "Zone 02", (3, 5, 7, 3, 1, 1), (1, 5, 7, 1, 1, 1), (1, 2, 3))
ee(a)

tt("# 06# Boundaries", '')
a.bcwrite(1, 1, "I low", CGK.BCTunnelInflow, CGK.PointRange, [(1, 1, 1), (3, 2, 4)])
ee(a)
a.bcdatasetwrite(1, 1, 1, "I low DATA SET", CGK.BCTunnelInflow)
ee(a)
a.bcdatawrite(1, 1, 1, 1, CGK.Neumann)
ee(a)

tt("# 07# Reference state (+ Data arrays)", '')
a.goto(1, [])
ee(a)
a.statewrite("ReferenceState")
ee(a)
a.goto(1, [(CGK.ReferenceState_t, 1)])
ee(a)
v = N.array([0.8], 'd')
a.arraywrite("Mach", CGK.RealDouble, 1, (1,), v)
ee(a)

tt("# 08# Equations (goto)", '')
a.goto(1, [(CGK.Zone_t, 1)])
ee(a)
a.equationsetwrite(5)
ee(a)
a.goto(1, [(CGK.Zone_t, 1), (CGK.FlowEquationSet_t, 1)])
ee(a)
a.governingwrite(CGK.NSTurbulent)
ee(a)
a.goto(1, [(CGK.Zone_t, 1), (CGK.FlowEquationSet_t, 1), (CGK.GoverningEquations_t, 1)])
ee(a)
a.diffusionwrite((1, 1, 1, 1, 1, 1))
ee(a)
a.goto(1, [(CGK.Zone_t, 1), (CGK.FlowEquationSet_t, 1)])
ee(a)
a.modelwrite(CGK.GasModel_t, CGK.Ideal)
ee(a)
a.modelwrite(CGK.ViscosityModel_t, CGK.SutherlandLaw)
ee(a)
a.modelwrite(CGK.TurbulenceModel_t, CGK.UserDefined)
ee(a)
a.simulationtypewrite(1, CGK.NonTimeAccurate)
ee(a)

tt("# 09# User defined data (goto)", '')
a.goto(1, [])
ee(a)
a.userdatawrite("Attributes")
ee(a)

tt("# 10# Descriptions (goto)", '')
a.goto(1, [(CGK.UserDefinedData_t, 1)])
ee(a)
a.descriptorwrite('.Title', 'pyCGNS: MLL extensive test')
ee(a)

tt("# 11# ", '')

tt("# 12# Solutions", '')
a.solwrite(1, 1, "Initialize", CGK.CellCenter)
ee(a)
a.solwrite(1, 1, "Result", CGK.CellCenter)
ee(a)
a.solwrite(1, 2, "Initialize", CGK.CellCenter)
ee(a)
a.solwrite(1, 2, "Result", CGK.CellCenter)
ee(a)
a.solwrite(1, 3, "Initialize", CGK.CellCenter)
ee(a)
a.solwrite(1, 3, "Result", CGK.CellCenter)
ee(a)
for s in [1, 2]:
    for z in [1, 2, 3]:
        for f in ["Density", "Pressure", "Temperature"]:
            a.fieldwrite(1, z, s, CGK.RealDouble, f, d[f])
            ee(a)

tt("# 13# Dataclass/ Exponents/ Dimensional", '')
a.goto(1, [(CGK.Zone_t, 1), (CGK.FlowSolution_t, 2)])
ee(a)
a.dataclasswrite(CGK.NormalizedByDimensional)
ee(a)
a.unitswrite(CGK.Kilogram, CGK.Meter, CGK.Second, CGK.Kelvin, CGK.Radian)
ee(a)
a.goto(1, [(CGK.ReferenceState_t, 1)])
ee(a)
v = N.array([2.718], 'd')
# A lot of non-sense data in there, try to set all of them without crash !
a.arraywrite("Nonsense value", CGK.RealDouble, 1, (1,), v)
ee(a)
a.goto(1, [(CGK.ReferenceState_t, 1), (CGK.DataArray_t, 1)])
ee(a)
a.dataclasswrite(CGK.NormalizedByDimensional)
ee(a)
a.unitswrite(CGK.Slug, CGK.Foot, CGK.UserDefined, CGK.Rankine, CGK.Null)
ee(a)
a.exponentswrite(CGK.RealDouble, (0, 1, -1, 0, 0))
ee(a)
a.conversionwrite(CGK.RealDouble, (0, 0))
ee(a)

tt("# 14# Convergence", '')
a.goto(1, [])
ee(a)
a.convergencewrite(2000, "GlobalConvergenceHistory")
ee(a)

tt("# 15# Family", '')
a.familywrite(1, "LeftPartOfThing")
ee(a)
a.goto(1, [(CGK.Zone_t, 1)])
ee(a)
a.familynamewrite("LeftPartOfThing")
ee(a)
a.goto(1, [(CGK.Zone_t, 2)])
ee(a)
a.familynamewrite("LeftPartOfThing")
ee(a)

tt("# 16# Ordinal(s)", '')
a.goto(1, [(CGK.Family_t, 1)])
ee(a)
a.ordinalwrite(628)
ee(a)

tt("# 17# Discrete", '')
a.discretewrite(1, 1, "Fluxes")
ee(a)

tt("# 18# Integral", '')
a.goto(1, [(CGK.Family_t, 1)])
ee(a, 3)
a.goto(1, [(CGK.Zone_t, 1)])
ee(a)
a.integralwrite(CGK.CoefLift)
ee(a)

tt("# 19# Grid location", '')
a.goto(1, [(CGK.Zone_t, 1), (CGK.DiscreteData_t, 1)])
ee(a)
a.gridlocationwrite(CGK.CellCenter)
ee(a)

tt("# 20# Spare Rinds", '')
a.goto(1, [(CGK.Zone_t, 1), (CGK.DiscreteData_t, 1)])
ee(a)
a.rindwrite((1, 2, 3, 4, 5, 6))
ee(a)

tt("# 21# Grid node", '')
a.gridwrite(1, 1, "ALE step 023")
ee(a)

tt("# 22# Geometry", '')
a.geowrite(1, 1, "Left Wing Red Light Bolt Number 23", "lw234_bolt.cad", "C4D")
ee(a, 1)
a.geowrite(1, 1, "Left Wing Red Light", "lw234.cad", "C4D")
ee(a)

tt("# 23# Part", '')
a.partwrite(1, 1, 1, "Left Red Light Bolt Number 23")
ee(a)

tt("# 24# Family BC", '')
a.familybocowrite(1, 1, "BoltsAndNuts", CGK.FamilySpecified)
ee(a)

tt("# 25# Iterative data", '')
a.biterwrite(1, "base iterative data", 10000)
ee(a)
a.ziterwrite(1, 1, "zone iterative data")
ee(a)

tt("# 26# Close (and del)", '')
a.close()
del a

tt("# 27# ADF ids", '')
# DB should be closed before, otherwise ids are not known !
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
print("#   #", a.root)
# print( "#   #", a.id(1,[(CGK.Zone_t,1),(CGK.FlowSolution_t,2)]))
print("#   #", a.baseid(1))
print("#   #", a.zoneid(1, 2))
print("#   #", a.coordid(1, 2, 1))
print("#   #", a.solid(1, 2, 1))
print("#   #", a.fieldid(1, 2, 1, 1))
print("#   #", a.one2oneid(1, 2, 1))
print("#   #", a.bcid(1, 1, 1))
ee(a)

tt("# 28# ADF reads", '')
# b=Mll.pyADF(a.root,"","")
# print "#   #", b.nodeAsDict(a.solid(1,2,1))
# print "#   #", b.nodeAsDict(a.fieldid(1,2,1,1))

tt("# 29# ADF close (MLL Still open)", '')
# b.database_close()

tt("# 30# MLL close", '')
a.close()

tt("# 31# Reopen", '')
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
ee(a)

tt("# 32# Link same file", '')
a.goto(1, [])
ee(a)
a.linkwrite("LinkToLocalZone", "", "/Base/Zone 03")
ee(a)
a.close()
ee(a)

tt("# 33# Link other file", '')
b = Mll.pyCGNS(TDBNAME3, CGNS.WRA.MODE_WRITE)
ee(b)
b.basewrite("DumbBase", 3, 3)
ee(b)
b.zonewrite(1, "DumbZone", [2, 4, 8, 1, 3, 7, 0, 0, 0], CGK.Structured)
ee(b)
b.close()
ee(b)

a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
ee(a)
a.goto(1, [])
ee(a)
a.linkwrite("LinkToExternalZone", TDBNAME3, "/DumbBase/DumbZone")
ee(a)
a.close()
ee(a)

tt("# 34# Close/ Reopen", '')
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
ee(a)
a.close()
ee(a)

tt("# 35# Get link infos", '')
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
for n in range(1, a.nbases() + 1):
    a.goto(1, [(CGK.Zone_t, n)])
    if (a.islink()):
        print("#   #", a.linkread())
a.close()

# ADF open of MLL file doesnot work anymore ;(

# tt("# 36# Delete link nodes (large stuff in there)",'')
# print "#   #", "Re-open as ADF file"
# b=Mll.pyADF(a.root,"","")
# print "#   #", "Search for links (MLL)"
# namelist=[]
# for n in range(1,a.nzones(1)+1):
#   a.goto(1,[(CGK.Zone_t,n)])
#   ee(a)
#   if (a.islink()):
#     print "#   #", "Get link ADF id (MLL)"
#     idlink=a.id(1,[(CGK.Zone_t,n)])
#     print "#   #", "Get node name (ADF)"
#     namelist+=[b.nodeAsDict(idlink)['name']]
# print "#   #", "Close ADF (shadow) file"
# b.database_close()

# a.goto(1,[])
# for name in namelist:    
#     print "#   #", "Delete node (MLL):", name
#     a.deletenode(name)
#     ee(a)

tt("# 37# Close/ Reopen", '')
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
a.close()
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_MODIFY)
ee(a)

tt("# 38# Arbitrary Grid Motion", '')
a.arbitrarymotionwrite(1, 1, "Rotor", CGK.NonDeformingGrid)
ee(a)

tt("# 39# Gravity", '')
a.gravitywrite(1, (0.1, 0.2, 0.3))
ee(a)

tt("# 40# Axisymmetry", '')
a.axisymwrite(1, (0.1, 0.2, 0.3), (0.1, 0.2, 0.3))  # should fail 3D !
ee(a, 1)

tt("# 41# Rigid motion (Data array with mandatory name)", '')
a.rigidmotionwrite(1, 1, "Rigid", CGK.UserDefined)
ee(a, 1)
a.goto(1, [(CGK.Zone_t, 1), (CGK.RigidGridMotion_t, 1)])
# very strange: [X before, X after], [Y before, Y after]...
# instead of [point before, point after] and point [x,y,z] !
w = N.array([[0.891, 4.12], [1.0, 2.2], [3.14159, 3.2]], 'd')
a.arraywrite("OriginLocation", CGK.RealDouble, len(w.shape), w.shape, w)
ee(a)

tt("# 42# Rotating", '')
a.goto(1, [])
a.rotatingwrite((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
ee(a)

tt("# 43# BC Wall Function", '')
a.bcwrite(1, 1, "I low", CGK.BCTunnelInflow, CGK.PointRange, [(1, 1, 1), (3, 2, 4)])
a.bcwallfunctionwrite(1, 1, 1, CGK.Generic)
ee(a)

tt("# 44# BC Area", '')
a.bcareawrite(1, 1, 1, CGK.BleedArea, 3.14159, "Region")
ee(a)

tt("# 45# BC Normal", '')
w = N.arange(3 * 24, dtype=N.float64)
w = N.reshape(w, (3, 24))
a.bcnormalwrite(1, 1, 1, (1, 2, 3), 1, CGK.RealDouble, w)
ee(a)

tt("# 46# Overset Holes", '')
w = N.array([[4, 2, 2], [4, 2, 3], [4, 4, 9]], 'i')
a.holewrite(1, 1, "Bomb", CGK.CellCenter, CGK.PointList, w)
ee(a)

tt("# 47# Generalized Connectivity", '')
ptc = N.array([[4, 2, 2], [4, 2, 3], [4, 4, 9]], 'i')
ptd = N.array([[4, 1, 2], [2, 2, 3], [5, 4, 9]], 'i')
cid1 = a.connwrite(1, 1, "Zone 01-Zone 03",
                   CGK.CellCenter, CGK.Abutting, CGK.PointList,
                   3, ptc, "Zone 03", CGK.Structured, CGK.PointListDonor,
                   CGK.Integer, 3, ptd)
ee(a)
pte = N.array([[4, 2, 5], [1, 2, 3], [4, 0, 9]], 'i')
ptf = N.array([[4, 1, 5], [1, 2, 3], [5, 0, 9]], 'i')
cid2 = a.connwrite(1, 1, "Zone 01-Zone 03",
                   CGK.CellCenter, CGK.Abutting, CGK.PointList,
                   3, ptc, "Zone 03", CGK.Structured, CGK.PointListDonor,
                   CGK.Integer, 3, ptd)
ee(a)

tt("# 48# Average Connectivity", '')
a.connaveragewrite(1, 1, 1, CGK.AverageCircumferential)
ee(a)

tt("# 49# Periodic Connectivity", '')
a.connperiodicwrite(1, 1, 1, (2.3, 1.2, 32.1), (1.2, 3.4, 5.6), (7.54, 4.2, 44.3))
ee(a)

tt("# 50# Element Connectivity (large)", '')
print("#   #", "Create zone")
uz = a.zonewrite(1, "Zone 99", [3 * 5 * 7, 2 * 4 * 6, 0], CGK.Unstructured)
ee(a)
print("#   #", "Add coordinates")
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateX, c01[0])
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateY, c01[1])
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateZ, c01[2])
ee(a)
print("#   #", "Section write")
ar = N.ones(shape=(384,), dtype='i')
us = a.sectionwrite(1, uz, "Section 99", CGK.HEXA_8, 1, 2 * 4 * 6, 0, ar)
print("#   #", us)
ee(a)
print("#   #", "Parent data write")
# a.parentdatawrite(1,uz,us,...) ?
ee(a)

tt("# 90# Ordering dimensions", '')
a.close()
a = Mll.pyCGNS(TDBNAME4, CGNS.WRA.MODE_WRITE)
ee(a)
o = N.ones((1, 3, 5), 'd')
o[0, 2] = 2.
o[:, :, 3] = 3.
t = o.copy()
o.transpose()
a.basewrite('Base', 3, 3)
ee(a)
a.zonewrite(1, 'Zone', [1, 3, 5, 0, 2, 4, 0, 0, 0], CGK.Structured)
ee(a)
a.zonewrite(1, 'Enoz', [5, 3, 1, 4, 2, 0, 0, 0, 0], CGK.Structured)
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateX, o)
ee(a)
a.coordwrite(1, 2, CGK.RealDouble, CGK.CoordinateX, o)
ee(a)
a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateY, t)
ee(a)
a.coordwrite(1, 2, CGK.RealDouble, CGK.CoordinateY, t)
ee(a)
a.close()
a = Mll.pyCGNS(TDBNAME4, CGNS.WRA.MODE_READ)
xo = a.coordread(1, 1, CGK.CoordinateX)
print("#   # 1 = ", N.all(N.equal(o, xo)), xo.shape)
ee(a)
xo = a.coordread(1, 2, CGK.CoordinateX)
print("#   # 0 = ", N.all(N.equal(o, xo)), xo.shape)
ee(a)
xt = a.coordread(1, 1, CGK.CoordinateY)
print("#   # 0 = ", N.all(N.equal(t, xt)), xt.shape)
ee(a)
xt = a.coordread(1, 2, CGK.CoordinateY)
print("#   # 1 = ", N.all(N.equal(t, xt)), xt.shape)
ee(a)
a.close()

# to be done...
tt("# 91# Fortran/C dimensions ordering", '')
tt("# 92# ", '')
tt("# 93# ", '')
tt("# 94# ", '')
tt("# 95# ", '')
tt("# 96# ", '')
tt("# 97# ", '')
tt("# 98# ", '')
tt("# 99# ", '')

tt("#100# Close/ Reopen", '')
a.close()
a = Mll.pyCGNS(TDBNAME2, CGNS.WRA.MODE_READ)
ee(a)

tt("#101# Version", '')
print("#   #", U.pnice(a.version()))

tt("#102# Base", '')
print("#   #", a.nbases())
print("#   #", a.bases())
print("#   #", a.baseread(1))
ee(a)

tt("#103# Zones", '')
print("#   #", a.nzones(1))
for n in range(1, a.nzones(1) + 1):
    print("#   #", CGK.ZoneType_[a.zonetype(1, n)], a.zoneread(1, n))

tt("#104# Coordinates", '')
print("#   #", a.ncoords(1, 3))
print("#   #", a.coordinfo(1, 3, 2))
print("#   #", a.coordread(1, 3, CGK.CoordinateX).shape)
print("#   #", a.coordread(1, 3, CGK.CoordinateX)[1][2][3])

tt("#105# Connectivity", '')
print("#   #", a.none2oneglobal(1))
print("#   #", a.none2one(1, 2))
print("#   #", a.none2one(1, 2))
# print "#   #", a.one2onereadglobal(1)
print("#   #", a.one2oneread(1, 2, 1))
print("#   #", a.one2oneread(1, 2, 4))

tt("#106# Boundaries", '')
print("#   #", a.nbc(1, 1))
print("#   #", a.bcinfo(1, 3, 1))
print("#   #", a.bcinfo(1, 1, 4))
print("#   #", a.bcread(1, 3, 1))
print("#   #", a.bcread(1, 1, 4))
print("#   #", a.bcdatasetread(1, 3, 1, 1))
print("#   #", a.bcdatasetread(1, 1, 1, 1))

tt("#107# Reference state (+ Data arrays)", '')
a.goto(1, [])
ee(a)
print("#   #", a.stateread())
a.goto(1, [(CGK.ReferenceState_t, 1)])
ee(a)
print("#   #", a.narrays())
ee(a)
print("#   #", a.arrayinfo(1))
ee(a)
print("#   #", a.arrayread(1))
ee(a)

tt("#108# Equations", '')

tt("#109# User defined data", '')
a.goto(1, [])
ee(a)
print("#   #", a.nuserdata())
print("#   #", a.userdataread(1))

tt("#110# Descriptions", '')
a.goto(1, [(CGK.UserDefinedData_t, 1)])
ee(a)
print("#   #", a.ndescriptors())
print("#   #", a.descriptors())
print("#   #", a.descriptorread(1))

tt("#112# Solutions", '')
print("#   #", a.nsols(1, 1))
print("#   #", a.nsols(1, 2))
print("#   #", a.nfields(1, 1, 1))
print("#   #", a.nfields(1, 2, 1))
print("#   #", a.solinfo(1, 1, 1))
print("#   #", a.solinfo(1, 2, 1))
print("#   #", a.solinfo(1, 2, 2))
print("#   #", a.fieldinfo(1, 1, 1, 1))
print("#   #", a.fieldinfo(1, 2, 2, 1))
print("#   #", a.fieldread(1, 2, 2, CGK.Density, CGK.RealDouble, [1, 1, 1], [2, 4, 6]).shape)
try:
    print("#   #", a.fieldread(1, 1, 2, CGK.Density, CGK.RealDouble, [1, 1, 1], [2, 4, 6]).shape)
except CGK.midlevel.error:
    print("Expected error: ", a.error)

tt("#113# Dataclass/ Exponents/ Dimensional", '')
a.goto(1, [(CGK.ReferenceState_t, 1), (CGK.DataArray_t, 1)])
ee(a)
print("#   #", a.dataclassread())
print("#   #", a.conversioninfo())
print("#   #", a.conversionread())
print("#   #", a.exponentsinfo())
print("#   #", a.exponentsread())
print("#   #", a.unitsread())

tt("#114# Convergence", '')
a.goto(1, [(CGK.Zone_t, 1)])  # zone (empty)
ee(a)
print("#   #", a.convergenceread())
ee(a, 2)
a.goto(1, [])  # base
ee(a)
print("#   #", a.convergenceread())

tt("#115# Family", '')
print("#   #", a.nfamilies(1))
a.familyread(1, 3)
ee(a, 1)
print("#   #", a.familyread(1, 1))
ee(a)
a.goto(1, [(CGK.Zone_t, 1)])
ee(a)
print("#   #", a.familynameread())

tt("#116# Ordinal", '')
a.goto(1, [(CGK.Family_t, 1)])
ee(a)
print("#   #", a.ordinalread())
ee(a)

tt("#117# Discrete", '')
print("#   #", a.ndiscrete(1, 1))
# print "#   #", a.discreteread(1,1,1)

tt("#118# Integral", '')
a.goto(1, [(CGK.Zone_t, 1)])
ee(a)
print("#   #", a.nintegrals())
ee(a)

tt("#119# Grid Location", '')

tt("#120# Rind", '')

tt("#121# Grid node", '')
print("#   #", a.ngrids(1, 1))
ee(a)
print("#   #", a.gridread(1, 1, 1))
print("#   #", a.gridread(1, 2, 1))
ee(a)

tt("#122# Geometry", '')
print("#   #", a.georead(1, 1, 1))

tt("#123# Part", '')
print("#   #", a.partread(1, 1, 1, 1))

tt("#124# Part", '')
print("#   #", a.familybocoread(1, 1, 1))

tt("#125# Iterative data", '')
print("#   #", a.biterread(1))
ee(a)
# print "#   #", a.ziterread(1,1)
ee(a)

tt("#138# Arbitrary Grid Motion", '')
print("#   #", a.narbitrarymotions(1, 1))
print("#   #", a.arbitrarymotionread(1, 1, 1))
ee(a, 1)

tt("#139# Gravity", '')
print("#   #", U.pnice(a.gravityread(1)))
ee(a)

tt("#140# Axisymmetry", '')
print("#   #", a.axisymread(1))
ee(a, 2)

tt("#141# Rigid motion", '')
print("#   #", a.nrigidmotions(1, 1))
print("#   #", a.rigidmotionread(1, 1, 1))
ee(a, 1)

tt("#142# Rotating", '')
a.goto(1, [])
print("#   #", U.pnice(a.rotatingread()))
ee(a)

tt("#143# BC Wall Function", '')
print("#   #", CGK.WallFunctionType_[a.bcwallfunctionread(1, 1, 1)])
ee(a, 2)

tt("#144# BC Area", '')
print("#   #", U.pnice(a.bcarearead(1, 1, 1)))
ee(a, 2)

tt("#145# BC Normal", '')
print
"#   #", U.pnice(a.bcread(1, 1, 1))
ee(a, 1)

tt("#146# Overset Holes", '')
print("#   #", a.nholes(1, 1))
print("#   #", a.holeinfo(1, 1, 1))
print("#   #", a.holeread(1, 1, 1))
ee(a, 1)

tt("#147# Generalized Connectivity", '')
print("#   #", a.nconns(1, 1))
print("#   #", a.conninfo(1, 1, 1))
print("#   #", a.connread(1, 1, 1))
ee(a, 1)

tt("#148# Average Connectivity", '')
print("#   #", a.connaverageread(1, 1, 1))
ee(a, 2)

tt("#149# Periodic Connectivity", '')
print("#   #", U.pnice(a.connperiodicread(1, 1, 1)))
ee(a, 2)

tt("#150# Element Connectivity ", '')
for zu in range(1, a.nzones(1) + 1):
    if a.zonetype(1, zu) == CGK.ZoneType[CGK.Unstructured_s]:
        print("#   #", a.nsections(1, zu))
        ee(a)
        print("#   #", a.elementdatasize(1, zu, 1))
        ee(a)
        print("#   #", a.npe(CGK.ElementType[CGK.HEXA_8_s]))
        ee(a)
        print("#   #", a.sectionread(1, zu, 1))
        ee(a)
        print("#   #", a.elementsread(1, zu, 1)[0].shape)
        ee(a)
a.close()

tt("#900# Tests that should not generate CORE DUMPs...", TDBNAME4)
a = Mll.pyCGNS(TDBNAME4, CGNS.WRA.MODE_WRITE)
ee(a)
a.basewrite('Base', 3, 3)
ee(a)
a.zonewrite(1, 'Zone', [19, 29, 39, 18, 28, 38, 0, 0, 0], CGK.Structured)
ee(a)

tt("#901# None as array")
try:
    a.coordwrite(1, 1, CGK.RealDouble, CGK.CoordinateX, None)
except CGK.midlevel.error as eee:
    print("#   # expected error:", eee)
ee(a)
a.goto(1, [])
ee(a)
try:
    a.arraywrite("None", CGK.RealDouble, 1, (1,), None)
except CGK.midlevel.error as eee:
    print("#   # expected error:", eee)
ee(a)

a.close()

# -------------------------------------------------------------------------
print("-" * 70)
print("End test suite")
# -------------------------------------------------------------------------
# last line
