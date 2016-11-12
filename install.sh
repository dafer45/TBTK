#!/bin/bash -l

cd hdf5
	bash fetchAndBuild.sh
	cd ..

cp Lib/makefile_without_CUDA Lib/makefile
while [[ $# > 0 ]]
do
	key=$1
	case $key in
		-CUDA)
			cp Lib/makefile_with_CUDA Lib/makefile
			shift
		;;
	esac
done

cd Lib/
	make
	cd ..

cd Tools
	./build.sh
	cd ..
