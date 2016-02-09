#CC = g++
#CFLAGS = -Wall -std=c++11 -fopenmp
CC = nvcc
CFLAGS = -std=c++11 --compiler-options "-fopenmp"

all:
#	$(CC) $(CFLAGS) src/main.cpp -o build/a.out -lTightBinding -lblas -llapack -lhdf5 -lhdf5_cpp
	$(CC) $(CFLAGS) src/main.cpp -I../../TBTK/calc/TightBindingLib/include -I../../hdf5/hdf5-build/include -L../../TBTK/calc/TightBindingLib/build -L../../hdf5/hdf5-build/hdf5/lib -o build/a.out -lTightBinding -lblas -llapack -lhdf5 -lhdf5_cpp -lfftw3

clean:
	rm -r build/*

