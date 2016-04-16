#CC = g++
#CFLAGS = -Wall -std=c++11 -fopenmp
CC = nvcc
CFLAGS = -std=c++11 --compiler-options "-fopenmp"

all:
	$(CC) $(CFLAGS) src/main.cpp -I$(TBTK_dir)/TBTK/calc/TightBindingLib/include -I$(TBTK_dir)/hdf5/hdf5-build/hdf5/include -L$(TBTK_dir)/TBTK/calc/TightBindingLib/build -L$(TBTK_dir)/hdf5/hdf5-build/hdf5/lib -o build/a.out -lTightBinding -lblas -llapack -lhdf5 -lhdf5_cpp

clean:
	rm -r build/*

