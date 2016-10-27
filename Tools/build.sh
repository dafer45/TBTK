#!/bin/bash -l

cd EstimateDOS
	make
	cd ..

cd CalculateDOS
	make
	cd ..

cp EstimateDOS/build/a.out bin/EstimateDOS
cp CalculateDOS/build/a.out bin/CalculateDOS
