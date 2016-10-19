#!/bin/bash -l

cd hdf5
	bash fetchAndBuild.sh
	cd ..

cp TBTK/calc/TightBindingLib/makefile_without_CUDA TBTK/calc/TightBindingLib/makefile
while [[ $# > 0 ]]
do
	key=$1
	case $key in
		-CUDA)
			cp TBTK/calc/TightBindingLib/makefile_with_CUDA TBTK/calc/TightBindingLib/makefile
			shift
		;;
	esac
done

cd TBTK/calc/TightBindingLib/
	make
	cd ../../..

cd Tools
	bash build.sh
	cd ..
