#!/bin/bash -l

cd Visualizer_build
	cmake ../Visualizer/
	make
	cd ..

cp Visualizer_build/dist/bin/TBTKVisualizer ../Tools/bin/TBTKVisualizer
