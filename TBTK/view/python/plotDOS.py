import h5py
import numpy
import matplotlib.pyplot
import matplotlib.axes
import scipy.ndimage.filters
import pylab
import sys

if(len(sys.argv) != 2):
	print "Error: Needs one argument for .hdf5-filename"
	exit(1)

filename = sys.argv[1]

file = h5py.File(filename, 'r');
dataset = file['DOS']

limits = dataset.attrs['UpLowLimits']
resolution = dataset.size
print "UpLowLimits: " + str(dataset.attrs['UpLowLimits'])
print "Resolution: " + str(resolution)

energies = numpy.linspace(limits[0], limits[1], resolution)

sigma = 3

fig = matplotlib.pyplot.figure()
matplotlib.pyplot.plot(energies, scipy.ndimage.filters.gaussian_filter1d(dataset, sigma))
#matplotlib.pyplot.show()

fig.savefig('figures/DOS.png')
