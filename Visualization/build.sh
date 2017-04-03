#!/bin/bash -l

cd Visualizer_build
	cmake ../Visualizer/
	make clean
	make
	cd ..

cp Visualizer_build/dist/bin/TBTKVisualizer ../Tools/bin/TBTKVisualizer

cd RayTracePlotter
	make clean
	make
	cd ..

cp RayTracePlotter/build/a.out ../Tools/bin/TBTKRayTracePlotter
