<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<!-- 
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCHEXC - $Id: cgns.sch 358 2004-04-21 12:03:57Z  $ 
# 
# See file COPYING in the root directory of this Python module source 
# tree for license information.
# 
-->
<sch:schema 
 xmlns:sch="http://www.ascc.net/xml/schematron" 
 xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
 xmlns:rng="http://relaxng.org/ns/structure/1.0">

  <!-- ==================================================================== -->
  <sch:pattern name="No nodes with same name">
    <sch:rule abstract="true" id="SameNameNodeCheck">
      <sch:assert
       test="not (@Name = following-sibling::*/@Name) and
             not (@Name = preceding-sibling::*/@Name) "
       diagnostics="E000" />
    </sch:rule>
  </sch:pattern>
    
  <!-- ==================================================================== -->
  <sch:pattern name="PointRange should be within DimensionValues">
    <sch:rule abstract="true" id="PointRangeCheckDimensionValues">
      <sch:assert
       test="
(
  (child::*[position()=1]/attribute::IndexDimension = 1)
  and
  ( substring-after(
      substring-before(
      ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),'[')
   >= substring-after(
      substring-before(
      child::*[position()=1]/attribute::Begin,']'),'['))
)
or 
(
  (child::*[position()=1]/attribute::IndexDimension = 2)
  and
  ( substring-after(
      substring-before(
      ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'[')
   >= substring-after(
      substring-before(
      child::*[position()=1]/attribute::Begin,','),'['))
  and
  ( substring-after(
      substring-before(
      ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),',')
   >= substring-after(
      substring-before(
      child::*[position()=1]/attribute::Begin,']'),','))
)
or 
(
  (child::*[position()=1]/attribute::IndexDimension = 3)
  and
  ( substring-after(
      substring-before(
      ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'[')
   >= substring-after(
      substring-before(
      child::*[position()=1]/attribute::Begin,','),'[')
  )
  and
  ( substring-before(
    substring(ancestor::Zone_t[position()=1]/attribute::VertexSize,
    string-length(substring-after(substring-before(
    ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'['))+3),',')
   >= substring-before(
    substring(child::*[position()=1]/attribute::Begin,
    string-length(substring-after(substring-before(
    child::*[position()=1]/attribute::Begin,','),'['))+3),',')
  )
  and
  ( substring-after(
      substring-after(
      substring-before(
      ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),','),',')
   >= substring-after(
      substring-after(
      substring-before(
      child::*[position()=1]/attribute::Begin,']'),','),',')
  )
)
or (child::*[position()=1]/attribute::IndexDimension > 3)
"
       diagnostics="E090" />
    </sch:rule>
  </sch:pattern>

  <!-- ==================================================================== -->
  <sch:pattern name="CGNS version check">
    <sch:rule context="CGNSTree">
      <extends rule="SameNameNodeCheck" />
      <sch:report test="@CGNSLibraryVersion != 2.4" 
		  diagnostics="W001" >
      </sch:report>
    </sch:rule>
  </sch:pattern>

  <!-- ==================================================================== -->
  <sch:pattern name="Consistency of base dimensions">
    <sch:rule context="CGNSBase_t">
      <extends rule="SameNameNodeCheck" />
      <sch:assert test="@PhysicalDimension >= @CellDimension" 
		  diagnostics="E001" />
    </sch:rule>
  </sch:pattern>

  <!-- ==================================================================== -->
  <sch:pattern name="Check that the size of Zone is forced in sub-tree">
    <sch:rule context="GridCoordinates_t">
      <extends rule="SameNameNodeCheck" />
      <sch:assert test="@VertexSize = ../@VertexSize" 
		  diagnostics="E010" />
    </sch:rule>
    <sch:rule context="GridCoordinates_t/DataArray_t">
      <extends rule="SameNameNodeCheck" />
      <sch:assert test="../@VertexSize = @DimensionValues" 
		  diagnostics="E011"/>
    </sch:rule>
  </sch:pattern>

  <!-- ==================================================================== -->
  <sch:pattern name="Grid Connectivity">
    <sch:rule context="GridConnectivity1to1_t">
       <extends rule="SameNameNodeCheck" />
       <extends rule="PointRangeCheckDimensionValues" />
    </sch:rule>
  </sch:pattern>

  <!-- ==================================================================== -->
  <sch:diagnostics>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
    <sch:diagnostic id="W001">
       CGNS version is {<sch:value-of select="@CGNSLibraryVersion" />} 
       compiler is expecting 2.3
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
    <sch:diagnostic id="E000">
       Cannot have same name 
       {<sch:value-of select="attribute::Name" />} for nodes at same level.
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
    <sch:diagnostic id="E001">
       Cell dimension of base {<sch:value-of select="@Name" />} is
       {<sch:value-of select="@CellDimension" />} and physical dimension
       is {<sch:value-of select="@PhysicalDimension" />}.
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
     <sch:diagnostic id="E010">
       Size of Zone {<sch:value-of select="../@Name" />} 
       is <sch:value-of select="../@VertexSize"/> 
       and size of Grid {<sch:value-of select="@Name" />}
       is {<sch:value-of select="@VertexSize"/>}.
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
    <sch:diagnostic id="E011">
       Size of Grid {<sch:value-of select="../@Name" />} 
       is <sch:value-of select="../@VertexSize"/> 
       and size of Array {<sch:value-of select="@Name" />}
       is {<sch:value-of select="@DimensionValues"/>}.
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
    <sch:diagnostic id="E090">
      PointRange dimension in GridConnectivity 
      {<sch:value-of select="@Name" />} is
      {<sch:value-of select="child::*[position()=1]/attribute::Begin" />}
      which is not within zone 
      {<sch:value-of select="ancestor::Zone_t[position()=1]/attribute::Name"/>}
      dimension 
      {<sch:value-of 
      select="ancestor::Zone_t[position()=1]/attribute::VertexSize"/>}
    </sch:diagnostic>
    <!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->
  </sch:diagnostics>
  
</sch:schema>
