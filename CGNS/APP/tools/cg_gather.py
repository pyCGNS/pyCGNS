#!/usr/bin/env python
#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.MAP as CGM

"""
  cg_gather [options] 

"""
import argparse


def parse():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()


# --- last line
