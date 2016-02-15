## @package plotDOS
#  @brief Plot density of states
#
#  @author Kristofer Björnson

import h5py
import numpy
import matplotlib.pyplot
import matplotlib.axes
import scipy.ndimage.filters
import pylab
import sys

if(len(sys.argv) != 3):
	print "Error, the following arguments are needed: .hdf5-file, sigma"
	exit(1)

filename = sys.argv[1]
sigma = float(sys.argv[2])

file = h5py.File(filename, 'r');
dataset = file['DOS']

limits = dataset.attrs['UpLowLimits']
resolution = dataset.size
print "UpLowLimits: " + str(dataset.attrs['UpLowLimits'])
print "Resolution: " + str(resolution)

energies = numpy.linspace(limits[1], limits[0], resolution)
sigma_discrete_units = sigma*resolution/(limits[0] - limits[1])

fig = matplotlib.pyplot.figure()
matplotlib.pyplot.plot(energies, scipy.ndimage.filters.gaussian_filter1d(dataset, sigma_discrete_units))
matplotlib.pyplot.xlabel('E');
matplotlib.pyplot.ylabel('DOS');

fig.savefig('figures/DOS.png')
