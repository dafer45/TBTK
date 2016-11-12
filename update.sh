#!/bin/bash -l

cd Lib/
	make clean
	make
	cd ..

cd Tools
	./build.sh
	cd ..
