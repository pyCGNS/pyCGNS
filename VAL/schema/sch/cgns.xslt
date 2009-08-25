<?xml version="1.0" standalone="yes"?>
<axsl:stylesheet xmlns:axsl="http://www.w3.org/1999/XSL/Transform" xmlns:sch="http://www.ascc.net/xml/schematron" version="1.0">
  <axsl:output xmlns="http://www.w3.org/1999/xhtml" method="text"/>
  <axsl:template match="*|@*" mode="schematron-get-full-path">
    <axsl:apply-templates select="parent::*" mode="schematron-get-full-path"/>
    <axsl:text>/</axsl:text>
    <axsl:if test="count(. | ../@*) = count(../@*)">@</axsl:if>
    <axsl:value-of select="name()"/>
    <axsl:text>[</axsl:text>
    <axsl:value-of select="1+count(preceding-sibling::*[name()=name(current())])"/>
    <axsl:text>]</axsl:text>
  </axsl:template>
  <axsl:template match="/" mode="generate-id-from-path"/>
  <axsl:template match="text()" mode="generate-id-from-path">
    <axsl:apply-templates select="parent::*" mode="generate-id-from-path"/>
    <axsl:value-of select="concat('.text-', 1+count(preceding-sibling::text()), '-')"/>
  </axsl:template>
  <axsl:template match="comment()" mode="generate-id-from-path">
    <axsl:apply-templates select="parent::*" mode="generate-id-from-path"/>
    <axsl:value-of select="concat('.comment-', 1+count(preceding-sibling::comment()), '-')"/>
  </axsl:template>
  <axsl:template match="processing-instruction()" mode="generate-id-from-path">
    <axsl:apply-templates select="parent::*" mode="generate-id-from-path"/>
    <axsl:value-of select="concat('.processing-instruction-', 1+count(preceding-sibling::processing-instruction()), '-')"/>
  </axsl:template>
  <axsl:template match="@*" mode="generate-id-from-path">
    <axsl:apply-templates select="parent::*" mode="generate-id-from-path"/>
    <axsl:value-of select="concat('.@', name())"/>
  </axsl:template>
  <axsl:template match="*" mode="generate-id-from-path" priority="-0.5">
    <axsl:apply-templates select="parent::*" mode="generate-id-from-path"/>
    <axsl:text>.</axsl:text>
    <axsl:choose>
      <axsl:when test="count(. | ../namespace::*) = count(../namespace::*)">
        <axsl:value-of select="concat('.namespace::-',1+count(namespace::*),'-')"/>
      </axsl:when>
      <axsl:otherwise>
        <axsl:value-of select="concat('.',name(),'-',1+count(preceding-sibling::*[name()=name(current())]),'-')"/>
      </axsl:otherwise>
    </axsl:choose>
  </axsl:template>
  <axsl:template match="/">
    <axsl:apply-templates select="/" mode="M0"/>
    <axsl:apply-templates select="/" mode="M1"/>
    <axsl:apply-templates select="/" mode="M2"/>
    <axsl:apply-templates select="/" mode="M3"/>
    <axsl:apply-templates select="/" mode="M4"/>
    <axsl:apply-templates select="/" mode="M5"/>
  </axsl:template>
  <axsl:template match="text()" priority="-1" mode="M0"/>
  <axsl:template match="text()" priority="-1" mode="M1"/>
  <axsl:template match="CGNSTree" priority="4000" mode="M2">
    <axsl:choose>
      <axsl:when test="not (@Name = following-sibling::*/@Name) and              not (@Name = preceding-sibling::*/@Name) "/>
      <axsl:otherwise>*** cgns error: [E000] Cannot have same name {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} for nodes at same level.
</axsl:otherwise>
    </axsl:choose>
    <axsl:if test="@CGNSLibraryVersion != 2.3">*** cgns warning: [W001] CGNS version is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@CGNSLibraryVersion"/><axsl:text xml:space="preserve"> </axsl:text>} compiler is expecting 2.3
</axsl:if>
    <axsl:apply-templates mode="M2"/>
  </axsl:template>
  <axsl:template match="text()" priority="-1" mode="M2"/>
  <axsl:template match="CGNSBase_t" priority="4000" mode="M3">
    <axsl:choose>
      <axsl:when test="not (@Name = following-sibling::*/@Name) and              not (@Name = preceding-sibling::*/@Name) "/>
      <axsl:otherwise>*** cgns error: [E000] Cannot have same name {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} for nodes at same level.
</axsl:otherwise>
    </axsl:choose>
    <axsl:choose>
      <axsl:when test="@PhysicalDimension &gt;= @CellDimension"/>
      <axsl:otherwise>*** cgns error: [E001] Cell dimension of base {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@CellDimension"/><axsl:text xml:space="preserve"> </axsl:text>} and physical dimension is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@PhysicalDimension"/><axsl:text xml:space="preserve"> </axsl:text>}.
</axsl:otherwise>
    </axsl:choose>
    <axsl:apply-templates mode="M3"/>
  </axsl:template>
  <axsl:template match="text()" priority="-1" mode="M3"/>
  <axsl:template match="GridCoordinates_t" priority="4000" mode="M4">
    <axsl:choose>
      <axsl:when test="not (@Name = following-sibling::*/@Name) and              not (@Name = preceding-sibling::*/@Name) "/>
      <axsl:otherwise>*** cgns error: [E000] Cannot have same name {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} for nodes at same level.
</axsl:otherwise>
    </axsl:choose>
    <axsl:choose>
      <axsl:when test="@VertexSize = ../@VertexSize"/>
      <axsl:otherwise>*** cgns error: [E010] Size of Zone {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="../@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="../@VertexSize"/><axsl:text xml:space="preserve"> </axsl:text>and size of Grid {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@VertexSize"/><axsl:text xml:space="preserve"> </axsl:text>}.
</axsl:otherwise>
    </axsl:choose>
    <axsl:apply-templates mode="M4"/>
  </axsl:template>
  <axsl:template match="GridCoordinates_t/DataArray_t" priority="3999" mode="M4">
    <axsl:choose>
      <axsl:when test="not (@Name = following-sibling::*/@Name) and              not (@Name = preceding-sibling::*/@Name) "/>
      <axsl:otherwise>*** cgns error: [E000] Cannot have same name {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} for nodes at same level.
</axsl:otherwise>
    </axsl:choose>
    <axsl:choose>
      <axsl:when test="../@VertexSize = @DimensionValues"/>
      <axsl:otherwise>*** cgns error: [E011] Size of Grid {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="../@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="../@VertexSize"/><axsl:text xml:space="preserve"> </axsl:text>and size of Array {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@DimensionValues"/><axsl:text xml:space="preserve"> </axsl:text>}.
</axsl:otherwise>
    </axsl:choose>
    <axsl:apply-templates mode="M4"/>
  </axsl:template>
  <axsl:template match="text()" priority="-1" mode="M4"/>
  <axsl:template match="GridConnectivity1to1_t" priority="4000" mode="M5">
    <axsl:choose>
      <axsl:when test="not (@Name = following-sibling::*/@Name) and              not (@Name = preceding-sibling::*/@Name) "/>
      <axsl:otherwise>*** cgns error: [E000] Cannot have same name {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} for nodes at same level.
</axsl:otherwise>
    </axsl:choose>
    <axsl:choose>
      <axsl:when test=" (   (child::*[position()=1]/attribute::IndexDimension = 1)   and   ( substring-after(       substring-before(       ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),'[')    &gt;= substring-after(       substring-before(       child::*[position()=1]/attribute::Begin,']'),'[')) ) or  (   (child::*[position()=1]/attribute::IndexDimension = 2)   and   ( substring-after(       substring-before(       ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'[')    &gt;= substring-after(       substring-before(       child::*[position()=1]/attribute::Begin,','),'['))   and   ( substring-after(       substring-before(       ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),',')    &gt;= substring-after(       substring-before(       child::*[position()=1]/attribute::Begin,']'),',')) ) or  (   (child::*[position()=1]/attribute::IndexDimension = 3)   and   ( substring-after(       substring-before(       ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'[')    &gt;= substring-after(       substring-before(       child::*[position()=1]/attribute::Begin,','),'[')   )   and   ( substring-before(     substring(ancestor::Zone_t[position()=1]/attribute::VertexSize,     string-length(substring-after(substring-before(     ancestor::Zone_t[position()=1]/attribute::VertexSize,','),'['))+3),',')    &gt;= substring-before(     substring(child::*[position()=1]/attribute::Begin,     string-length(substring-after(substring-before(     child::*[position()=1]/attribute::Begin,','),'['))+3),',')   )   and   ( substring-after(       substring-after(       substring-before(       ancestor::Zone_t[position()=1]/attribute::VertexSize,']'),','),',')    &gt;= substring-after(       substring-after(       substring-before(       child::*[position()=1]/attribute::Begin,']'),','),',')   ) ) or (child::*[position()=1]/attribute::IndexDimension &gt; 3) "/>
      <axsl:otherwise>*** cgns error: [E090] PointRange dimension in GridConnectivity {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="@Name"/><axsl:text xml:space="preserve"> </axsl:text>} is {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="child::*[position()=1]/attribute::Begin"/><axsl:text xml:space="preserve"> </axsl:text>} which is not within zone {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="ancestor::Zone_t[position()=1]/attribute::Name"/><axsl:text xml:space="preserve"> </axsl:text>} dimension {<axsl:text xml:space="preserve"> </axsl:text><axsl:value-of select="ancestor::Zone_t[position()=1]/attribute::VertexSize"/><axsl:text xml:space="preserve"> </axsl:text>}
</axsl:otherwise>
    </axsl:choose>
    <axsl:apply-templates mode="M5"/>
  </axsl:template>
  <axsl:template match="text()" priority="-1" mode="M5"/>
  <axsl:template match="text()" priority="-1"/>
</axsl:stylesheet>
