#!/bin/bash -l

cd TBTK/calc/TightBindingLib/
	make clean
	make
	cd ../../..

cd Tools
	./build.sh
	cd ..
