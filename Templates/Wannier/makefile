##########################################################
#GCC compiler (comment out if using NVCC compiler below) #
##########################################################
CC = g++
CFLAGS = -Wall -std=c++11 -fopenmp -O3
LIBS = -lTBTK -lblas -llapack -lhdf5 -lhdf5_cpp

######################################################
#NVCC compiler (uncomment if CUDA support is needed) #
######################################################
#CC = nvcc
#CFLAGS = -std=c++11 --compiler-options "-Wall -fopenmp"
#LIBS = -lTBTK -lblas -llapack -lhdf5 -lhdf5_cpp

####################
#Main build target #
####################
all:
	@echo "Building..."
	@$(CC) $(CFLAGS) src/main.cpp -I$(TBTK_dir)/hdf5/hdf5-build/include -L$(TBTK_dir)/hdf5/hdf5-build/hdf5/lib -o build/a.out $(LIBS) `pkg-config --libs opencv`
	@echo "Done"

################
# Clean target #
################
clean:
	rm -r build/*

