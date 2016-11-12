# -*- encoding: utf-8 -*-

## @package TBTKview
#  @file plotDOS.py
#  @brief Plot density
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

if(len(sys.argv) != 2):
	print "Error, the following arguments are needed: .hdf5-file"
	exit(1)

filename = sys.argv[1]

file = h5py.File(filename, 'r');
dataset = file['Density']

dimensions = dataset.shape
print "Dimensions: " + str(dimensions)
if(len(dimensions) != 2):
	print "Error, can only plot for 2 physical dimensions"
	exit(0)

x = numpy.arange(0, dimensions[0], 1)
y = numpy.arange(0, dimensions[1], 1)
X, Y = numpy.meshgrid(x, y)

fig = matplotlib.pyplot.figure()
ax = fig.gca(projection='3d')

Z = dataset[:,:]
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(numpy.min(Z), numpy.max(Z))

fig.savefig('figures/Density.png')
