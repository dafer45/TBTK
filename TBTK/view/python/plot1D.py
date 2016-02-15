## @package plot1D
#  @brief Plot line
#
#  @author Kristofer Björnson

import h5py
import matplotlib.pyplot
import sys

if(len(sys.argv) != 3):
	print "Error, the following arguments are needed: .hdf5-file, dataset name"
	exit(1)

filename = sys.argv[1]
dataset_name = sys.argv[2]

file = h5py.File(filename, 'r');

dataset = file[dataset_name]

fig = matplotlib.pyplot.figure()
matplotlib.pyplot.plot(dataset)
fig.savefig('figures/' + dataset_name + '.png')
