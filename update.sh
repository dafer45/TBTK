#!/bin/bash -l

useCuda=false
while [[ $# > 0 ]]
do
	key=$1
	case $key in
		-CUDA)
			useCuda=true
			shift
		;;
	esac
done

echo "##################"
echo "# Rebuilding Lib #"
echo "##################"
cd Lib/
	make clean
	if $useCuda;
	then
		make cuda
	else
		make nocuda
	fi
	cd ..

#echo ""
#echo "####################"
#echo "# Rebuilding Tools #"
#echo "####################"
#cd Tools
#	./build.sh
#	cd ..
