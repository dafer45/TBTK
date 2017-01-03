#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## @package TBTKview
#  @file TBTKPlot2D.py
#  @brief Plot 2D
#
#  @author Kristofer Björnson

import h5py
import numpy
import matplotlib.pyplot
import matplotlib.axes
import matplotlib.cm
import scipy.ndimage.filters
import mpl_toolkits.mplot3d
import sys

if(len(sys.argv) != 3):
	print "Error, the following parameters are needed: .hdf5-file"
	exit(1)

filename = sys.argv[1]
dataset_name = sys.argv[2]

file = h5py.File(filename, 'r');
dataset = file[dataset_name]

size_x = dataset.shape[0]
size_y = dataset.shape[1]

x = numpy.arange(0, dataset.shape[0], 1)
y = numpy.arange(0, dataset.shape[1], 1)
X, Y = numpy.meshgrid(x, y)

fig = matplotlib.pyplot.figure()
Z = dataset[:,:]

#Color map figure
ax = fig.gca()
ax.pcolormesh(X.transpose(), Y.transpose(), Z, cmap=matplotlib.cm.coolwarm)

fig.savefig("figures/" + dataset_name + ".png")

