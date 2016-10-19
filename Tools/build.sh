#!/bin/bash -l

cd EstimateDOS
	make
	cd ..

cp EstimateDOS/build/a.out bin/EstimateDOS
