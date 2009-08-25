<?xml version="1.0" ?>
<!-- 
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCHEXC - $Id: cgns.xsl 132 2004-01-29 10:39:56Z  $ 
# 
# See file COPYING in the root directory of this Python module source 
# tree for license information.
# 
-->
<xsl:stylesheet
   version="1.0"
   xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
   xmlns:axsl="http://www.w3.org/1999/XSL/TransformAlias"
   xmlns="http://www.w3.org/1999/xhtml">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:apply-templates/>
  </xsl:copy>
</xsl:template>

<!-- <xsl:template match="@DimensionValues"> -->
<!--   <xsl:attribute name="DimensionValues"> -->
<!--     <xsl:value-of  -->
<!--      select="translate(substring-before(substring-after(.,'['),']'),',',' ')"/> -->
<!--   </xsl:attribute>  -->
<!-- </xsl:template> -->

</xsl:stylesheet>
