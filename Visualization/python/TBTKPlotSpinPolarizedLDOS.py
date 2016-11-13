#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## @package TBTKview
#  @file plotSP_LDOS.py
#  @brief Plot local density of states
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

if(len(sys.argv) != 5):
	print "Error, the following parameters are needed: .hdf5-file, theta, phi, sigma"
	exit(1)

filename = sys.argv[1]
theta = float(sys.argv[2])
phi = float(sys.argv[3])
sigma = float(sys.argv[4])

file = h5py.File(filename, 'r');
dataset = file['SpinPolarizedLDOS']

data_dimensions = dataset.shape
physical_dimensions = len(data_dimensions) - 3 #Three last dimensions are for energy, spin components, and real/imaginary decomposition.
energy_resolution = data_dimensions[physical_dimensions];
limits = dataset.attrs['UpLowLimits']
print "Dimensions: " + str(physical_dimensions)
print "Resolution: " + str(energy_resolution)
print "UpLowLimits: " + str(limits)
if(physical_dimensions != 1):
	print "Error, can only plot for 1 physical dimensions"
	exit(0)

size_x = data_dimensions[0]
size_y = data_dimensions[1]

x = numpy.arange(0, data_dimensions[0], 1)
y = numpy.arange(limits[1], limits[0], (limits[0] - limits[1])/energy_resolution)
X, Y = numpy.meshgrid(x, y)

fig = matplotlib.pyplot.figure()
Z = numpy.real((dataset[:,:,0,0] + 1j*dataset[:,:,0,1])*math.cos(theta/2)*math.cos(theta/2) \
		+ (dataset[:,:,1,0] + 1j*dataset[:,:,1,1])*math.cos(theta/2)*math.sin(theta/2)*cmath.exp(1j*phi) \
		+ (dataset[:,:,2,0] + 1j*dataset[:,:,2,1])*math.sin(theta/2)*math.cos(theta/2)*cmath.exp(-1j*phi) \
		+ (dataset[:,:,3,0] + 1j*dataset[:,:,3,1])*math.sin(theta/2)*math.sin(theta/2) \
)
sigma_discrete_units = sigma*energy_resolution/(limits[0] - limits[1])
for xp in range(0, size_x):
	Z[xp,:] = scipy.ndimage.filters.gaussian_filter1d(Z[xp,:], sigma_discrete_units)

#Color map figure
ax = fig.gca()
ax.pcolormesh(X.transpose(), Y.transpose(), Z, cmap=matplotlib.cm.coolwarm)

fig.savefig('figures/SpinPolarizedLDOS.png')

