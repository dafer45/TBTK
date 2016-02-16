import h5py
import numpy
import matplotlib.pyplot
import matplotlib.axes
import matplotlib.cm
import scipy.ndimage.filters
import mpl_toolkits.mplot3d
import sys

if(len(sys.argv) != 2):
	print "Error: Needs one argument for .hdf5-filename"
	exit(1)

filename = sys.argv[1]

file = h5py.File(filename, 'r');
dataset = file['SP_LDOS_E']

data_dimensions = dataset.shape
physical_dimensions = len(data_dimensions) - 2 #Two last dimensions are for energy and spin components.
energy_resolution = data_dimensions[physical_dimensions];
limits = dataset.attrs['UpLowLimits']
print "Dimensions: " + str(physical_dimensions)
print "Resolution: " + str(energy_resolution)
print "UpLowLimits: " + str(limits)
if(physical_dimensions != 1):
	print "Error, can only plot for 1 physical dimensions"
	exit(0)

x = numpy.arange(0, data_dimensions[0], 1)
y = numpy.arange(limits[1], limits[0], (limits[0] - limits[1])/energy_resolution)
X, Y = numpy.meshgrid(y, x)

for n in range(0, 5):
	fig = matplotlib.pyplot.figure()
	Z = dataset[:,:,n]

	#3D figure
#	ax = fig.gca(projection='3d')
#	ax.view_init(90, 0)
#	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
#	ax.set_zlim(numpy.min(Z), numpy.max(Z))

	#Color map figure
	ax = fig.gca()
	ax.pcolormesh(Y, X, Z, cmap=matplotlib.cm.coolwarm)

#	matplotlib.pyplot.show()
	fig.savefig('figures/SP_LDOS_E' + str(n) + '.png')

