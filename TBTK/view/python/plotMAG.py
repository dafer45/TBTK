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
dataset = file['MAG']

data_dimensions = dataset.shape
physical_dimensions = len(data_dimensions) - 1 #Last dimension for spin components.
print "Dimensions: " + str(physical_dimensions)
if(physical_dimensions != 2):
	print "Error, can only plot for 2 physical dimensions"
	exit(0)

x = numpy.arange(0, data_dimensions[0], 1)
y = numpy.arange(0, data_dimensions[1], 1)
X, Y = numpy.meshgrid(x, y)

for n in range(0, 3):
	fig = matplotlib.pyplot.figure()
	ax = fig.gca(projection='3d')

	Z = dataset[:,:,n]
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_zlim(numpy.min(Z), numpy.max(Z))
#	matplotlib.pyplot.show()
	fig.savefig('figures/MAG' + str(n) + '.png')

