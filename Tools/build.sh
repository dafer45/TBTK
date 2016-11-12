#!/bin/bash -l

cd EstimateDOS
	rm build/* 2> /dev/null
	make
	cd ..

cd CalculateDOS
	rm build/* 2> /dev/null
	make
	cd ..

cp EstimateDOS/build/a.out bin/TBTKEstimateDOS
cp CalculateDOS/build/a.out bin/TBTKCalculateDOS
