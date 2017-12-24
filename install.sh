#!/bin/bash -l

echo "#################################"
echo "# Downloading and building HDF5 #"
echo "#################################"
cd hdf5
	bash fetchAndBuild.sh
	cd ..

echo ""
echo "################"
echo "# Building Lib #"
echo "################"
#cp Lib/makefile_without_CUDA Lib/makefile
useCuda=false
while [[ $# > 0 ]]
do
	key=$1
	case $key in
		-CUDA)
#			cp Lib/makefile_with_CUDA Lib/makefile
			useCuda=true
			shift
		;;
	esac
done

cd Lib/
	if $useCuda;
	then
		make cuda
	else
		make nocuda
	fi
	cd ..

#cd Lib/
#	make
#	cd ..

#echo ""
#echo "##################"
#echo "# Building Tools #"
#echo "##################"
#cd Tools
#	./build.sh
#	cd ..
