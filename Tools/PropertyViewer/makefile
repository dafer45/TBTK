CC = g++
CFLAGS = -Wall -std=c++11 -fopenmp
#CC = nvcc
#CFLAGS = -std=c++11 --compiler-options "-fopenmp"

all:
	@echo "Building: TBTKEstimateDOS"
	@$(CC) $(CFLAGS) src/main.cpp -I$(TBTK_dir)/hdf5/hdf5-build/include -L$(TBTK_dir)/hdf5/hdf5-build/hdf5/lib -o build/a.out -lTBTK -llapack -lhdf5 -lhdf5_cpp -lblas -lcudart -lcusparse `wx-config --cxxflags` `wx-config --libs` `pkg-config --libs opencv`

clean:
	rm -r build/*

