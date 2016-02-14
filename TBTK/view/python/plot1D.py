import h5py
import matplotlib.pyplot
import sys

if(len(sys.argv) != 2):
	print "Error: Needs one argument for .hdf5-filename"
	exit(1)

filename = sys.argv[1]

file = h5py.File(filename, 'r');

dataset = file['EV']

fig = matplotlib.pyplot.figure()
matplotlib.pyplot.plot(dataset)
#matplotlib.pyplot.show()
fig.savefig('figures/EV.png')
