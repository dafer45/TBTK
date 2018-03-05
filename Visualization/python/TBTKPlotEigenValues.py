#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## @package TBTKview
#  @file plotEigenValues.py
#  @brief Plot eigenvalues
#
#  @author Kristofer Björnson

import h5py
import matplotlib.pyplot
import sys
import os

if(len(sys.argv) != 2):
	print "Error, the following arguments are needed: .hdf5-file"
	exit(1)

#os.system("python " + os.environ['TBTK_dir'] + "/Visualization/python/TBTKPlot1D.py " + sys.argv[1] + " EigenValues")
os.system("${CMAKE_INSTALL_PREFIX}/bin/TBTKPlot1D.py " + sys.argv[1] + " EigenValues")

