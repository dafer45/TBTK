CC = g++
CFLAGS = -Wall -std=c++11
#CC = nvcc
#CFLAGS = -std=c++11

all:
	$(CC) $(CFLAGS) src/main.cpp -o build/a.out -lTightBinding -lblas -llapack -lhdf5 -lhdf5_cpp

clean:
	rm -r build/*

