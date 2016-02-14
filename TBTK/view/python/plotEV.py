import h5py
import matplotlib.pyplot
import sys
import os

if(len(sys.argv) != 2):
	print "Error, the following arguments are needed: .hdf5-file"
	exit(1)

os.system("python " + os.environ['TBTK_dir'] + "/TBTK/view/python/plot1D.py " + sys.argv[1] + " EV")

