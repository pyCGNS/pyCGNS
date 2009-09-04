<?xml version="1.0" ?>
<!-- 
# CFD General Notation System - CGNS XML tools
# ONERA/DSNA - poinot@onera.fr - henaux@onera.fr
# pyCHEXC - $Id: cgns.xsl 389 2004-08-31 08:15:27Z  $ 
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

<xsl:import href="file://TOBESETATINSTALLTIME/skeleton1-6.xsl"/>
<xsl:param name="diagnose">yes</xsl:param>     

<xsl:template name="process-prolog">
   <axsl:output method="text" />
</xsl:template>

<!-- <xsl:template name="process-pattern"> -->
<!--    <xsl:param name="name" /> -->
<!--    <xsl:text>*** Error      : </xsl:text> -->
<!--    <xsl:value-of select="$name" /> -->
<!--    <xsl:text>&#10;</xsl:text> -->
<!-- </xsl:template> -->

<!-- <xsl:template name="process-rule"> -->
<!--    <xsl:param name="context" /> -->
<!--    <xsl:text>*** Context    : </xsl:text> -->
<!--    <xsl:value-of select="$context" /> -->
<!--    <xsl:text>&#10;</xsl:text> -->
<!-- </xsl:template> -->

<xsl:template name="process-assert">
   <xsl:param name="diagnostics" />
   <xsl:apply-templates mode="text"/>
<!--    <xsl:text>&#10;</xsl:text> -->
   <xsl:text>*** cgns assert ERROR: [</xsl:text>
   <xsl:value-of select="$diagnostics" />
   <xsl:text>] </xsl:text>
   <xsl:call-template name="diagnosticsSplit">
     <xsl:with-param name="str" select="$diagnostics"/>
   </xsl:call-template>
   <xsl:text>&#10;</xsl:text>
</xsl:template>

<xsl:template name="process-report">
   <xsl:param name="diagnostics" />
   <xsl:apply-templates mode="text"/>
<!--    <xsl:text>&#10;</xsl:text> -->
   <xsl:text>*** cgns report ERROR : [</xsl:text>
   <xsl:value-of select="$diagnostics" />
   <xsl:text>] </xsl:text>
   <xsl:call-template name="diagnosticsSplit">
     <xsl:with-param name="str" select="$diagnostics"/>
   </xsl:call-template>
   <xsl:text>&#10;</xsl:text>
</xsl:template>

</xsl:stylesheet>
