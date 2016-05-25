wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.16/src/hdf5-1.8.16.tar
tar -xvf hdf5-1.8.16.tar
rm hdf5-1.8.16.tar
mkdir hdf5-build
	CFLAGS="-std=c++11"
	cd hdf5-build; ../hdf5-1.8.16/configure --enable-cxx
	cd hdf5-build; make
	cd hdf5-build; make install
