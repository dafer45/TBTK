#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## @package TBTKview
#  @file plotMAG.py
#  @brief Plot magnetization
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
import math
import cmath

if(len(sys.argv) != 4):
	print "Error: Needs one argument for .hdf5-filename, theta, phi"
	exit(1)

filename = sys.argv[1]
theta = float(sys.argv[2])
phi = float(sys.argv[3])

file = h5py.File(filename, 'r');
dataset = file['Magnetization']

data_dimensions = dataset.shape
physical_dimensions = len(data_dimensions) - 2 #Last two dimension for matrix elements and real/imaginary decomposition.
print "Dimensions: " + str(physical_dimensions)
if(physical_dimensions != 2):
	print "Error, can only plot for 2 physical dimensions"
	exit(0)

size_x = data_dimensions[0]
size_y = data_dimensions[1]

x = numpy.arange(0, size_x, 1)
y = numpy.arange(0, size_y, 1)
X, Y = numpy.meshgrid(x, y)

#mag_real = dataset[:,:,:,0]
#mag_imag = dataset[:,:,:,1]

Z=numpy.zeros((size_x, size_y))
for xp in range(0,size_x):
	for yp in range(0, size_y):
		uu = dataset[xp,yp,0,0] + 1j*dataset[xp,yp,0,1]
		ud = dataset[xp,yp,1,0] + 1j*dataset[xp,yp,1,1]
		du = dataset[xp,yp,2,0] + 1j*dataset[xp,yp,2,1]
		dd = dataset[xp,yp,3,0] + 1j*dataset[xp,yp,3,1]
		Z[xp,yp] = numpy.real( \
			+ (ud + du)*math.sin(theta)*math.cos(phi) \
			- 1j*(ud - du)*math.sin(theta)*math.sin(phi) \
			+ (uu-dd)*math.cos(theta) \
		)

fig = matplotlib.pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X.transpose(), Y.transpose(), Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(numpy.min(Z), numpy.max(Z))
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('Magnetization');
fig.savefig('figures/MAG.png')

#for n in range(0, 3):
#	fig = matplotlib.pyplot.figure()
#	ax = fig.gca(projection='3d')

#	Z = dataset[:,:,n]
#	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
#	ax.set_zlim(numpy.min(Z), numpy.max(Z))
##	matplotlib.pyplot.show()
#	fig.savefig('figures/MAG' + str(n) + '.png')

