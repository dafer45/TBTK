#!/bin/bash -l

cd hdf5
	bash fetchAndBuild.sh
	cd ..

cd TBTK/calc/TightBindingLib/
	make
