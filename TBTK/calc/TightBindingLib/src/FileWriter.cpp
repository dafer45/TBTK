/** @file FileWriter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/FileWriter.h"
//#include "../include/PropertyExtractor.h"
#include <string>
#include <sstream>
#include <iostream>
#include <H5Cpp.h>
#include <fstream>

#ifndef H5_NO_NAMESPACE
	using namespace H5;
#endif

using namespace std;

namespace TBTK{

bool FileWriter::isInitialized = false;
string FileWriter::filename = "TBTKResults.h5";

void FileWriter::init(){
	if(isInitialized)
		return;

	try{
		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);
		file.close();
	}
	catch(FileIException error){
		H5File file(filename, H5F_ACC_EXCL);
		file.close();
	}

	isInitialized = true;
}

void FileWriter::writeModel(Model *model, string name, string path){
	init();

	stringstream ss;
	ss << name << "AmplitudeSet";

	writeAmplitudeSet(&(model->amplitudeSet), ss.str(), path);

	ss.str("");
	ss << name << "Geometry";
	writeGeometry(model->getGeometry(), ss.str(), path);
}

void FileWriter::writeAmplitudeSet(AmplitudeSet *amplitudeSet, string name, string path){
	init();

	complex<double> *amplitudes;
	int *indices;
	int numHoppingAmplitudes;
	int maxIndexSize;
	amplitudeSet->tabulate(&amplitudes, &indices, &numHoppingAmplitudes, &maxIndexSize);

	const int INDEX_RANK = 3;
	hsize_t indexDims[INDEX_RANK];
	indexDims[0] = numHoppingAmplitudes;
	indexDims[1] = 2; //Two indices per HoppingAmplitude
	indexDims[2] = maxIndexSize;
	const int AMPLITUDE_RANK = 1;
	hsize_t amplitudeDims[AMPLITUDE_RANK];
	amplitudeDims[0] = 2*numHoppingAmplitudes;	//2 because data is complex<double> interpreted as 2*double

	try{
		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name << "Indices";

		DataSpace dataspace = DataSpace(INDEX_RANK, indexDims);
		DataSet dataset = DataSet(file.createDataSet(ss.str(), PredType::STD_I32BE, dataspace));
		dataset.write(indices, PredType::NATIVE_INT);
		dataspace.close();
		dataset.close();

		ss.str("");
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name << "Amplitudes";

		dataspace = DataSpace(AMPLITUDE_RANK, amplitudeDims);
		dataset = DataSet(file.createDataSet(ss.str(), PredType::IEEE_F64BE, dataspace));
		dataset.write(amplitudes, PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeAmplitudeSet: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeAmplitudeSet: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeAmplitudeSet: While writing to " << name << "\n";
		exit(1);
	}

	delete [] amplitudes;
	delete [] indices;
}

void FileWriter::writeGeometry(const Geometry *geometry, string name, string path){
	init();

	int dimensions = geometry->getDimensions();
	int numSpecifiers = geometry->getNumSpecifiers();
	const double* coordinates = geometry->getCoordinates();
	const int* specifiers = geometry->getSpecifiers();
	int basisSize = geometry->getBasisSize();

	const int RANK = 2;
	hsize_t dDims[RANK];
	dDims[0] = basisSize;
	dDims[1] = dimensions;
	hsize_t sDims[RANK];
	sDims[0] = basisSize;
	sDims[1] = numSpecifiers;

	try{
		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name << "Coordinates";

		DataSpace dataspace = DataSpace(RANK, dDims);
		DataSet dataset = DataSet(file.createDataSet(ss.str(), PredType::IEEE_F64BE, dataspace));
		dataset.write(coordinates, PredType::NATIVE_DOUBLE);
		dataset.close();
		dataspace.close();

		ss.str("");
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name << "Specifiers";

		dataspace = DataSpace(RANK, sDims);
		dataset = DataSet(file.createDataSet(ss.str(), PredType::STD_I32BE, dataspace));
		dataset.write(specifiers, PredType::NATIVE_INT);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeGeometry: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeGeometry: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeGeometry: While writing to " << name << "\n";
		exit(1);
	}
}

/*void FileWriter::writeEigenValues(const double *ev, int size, string name, string path){
	init();

	const int RANK = 1;
	hsize_t dims[1];
	dims[0] = size;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(RANK, dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(ev, PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}
}*/

void FileWriter::writeEigenValues(const Property::EigenValues *ev, string name, string path){
	init();

	const int RANK = 1;
	hsize_t dims[1];
	dims[0] = ev->getSize();

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(RANK, dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(ev->getData(), PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeEigenValues: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeEigenValues: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeEigenValues: While writing to " << name << "\n";
		exit(1);
	}
}

/*void FileWriter::writeDOS(const double *dos, double l_lim, double u_lim, int resolution, string name, string path){
	init();

	const int DOS_RANK = 1;
	hsize_t dos_dims[1];
	dos_dims[0] = resolution;

	double limits[2];
	limits[0] = u_lim;
	limits[1] = l_lim;
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(DOS_RANK, dos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(dos, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}
}*/

void FileWriter::writeDOS(const Property::DOS *dos, string name, string path){
	init();

	const int DOS_RANK = 1;
	hsize_t dos_dims[1];
	dos_dims[0] = dos->getResolution();

	double limits[2];
	limits[0] = dos->getUpperBound();
	limits[1] = dos->getLowerBound();
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(DOS_RANK, dos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(dos->getData(), PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeDOS: While writing to " << name << "\n";
		exit(1);
	}
}

/*void FileWriter::writeDensity(const double *density, int rank, const int *dims, string name, string path){
	init();

	hsize_t density_dims[rank];
	for(int n = 0; n < rank; n++)
		density_dims[n] = dims[n];

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, density_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(density, PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();
		file.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}
}*/

void FileWriter::writeDensity(const Property::Density *density, string name, string path){
	init();

	int rank = density->getDimensions();
	const int *dims = density->getRanges();

	hsize_t density_dims[rank];
	for(int n = 0; n < rank; n++)
		density_dims[n] = dims[n];

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, density_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(density->getData(), PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();
		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeDensity: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeDensity: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeDensity: While writing to " << name << "\n";
		exit(1);
	}
}

/*void FileWriter::writeMAG(const complex<double> *mag, int rank, const int *dims, string name, string path){
	init();

	hsize_t mag_dims[rank+2];//Last two dimension for matrix elements and real/imaginary decomposition.
	for(int n = 0; n < rank; n++)
		mag_dims[n] = dims[n];
	const int NUM_MATRIX_ELEMENTS = 4;
	mag_dims[rank] = NUM_MATRIX_ELEMENTS;

	int mag_size = 1;
	for(int n = 0; n < rank+1; n++)
		mag_size *= mag_dims[n];
	double *mag_decomposed;
	mag_decomposed = new double[2*mag_size];
	for(int n = 0; n < mag_size; n++){
		mag_decomposed[2*n+0] = real(mag[n]);
		mag_decomposed[2*n+1] = imag(mag[n]);
	}
	mag_dims[rank+1] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+2, mag_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(mag_decomposed, PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();
		file.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}

	delete [] mag_decomposed;
}*/

void FileWriter::writeMagnetization(const Property::Magnetization *magnetization, string name, string path){
	init();

	int rank = magnetization->getDimensions();
	const int *dims = magnetization->getRanges();
	const complex<double> *data = magnetization->getData();

	hsize_t mag_dims[rank+2];//Last two dimension for matrix elements and real/imaginary decomposition.
	for(int n = 0; n < rank; n++)
		mag_dims[n] = dims[n];
	const int NUM_MATRIX_ELEMENTS = 4;
	mag_dims[rank] = NUM_MATRIX_ELEMENTS;

	int mag_size = 1;
	for(int n = 0; n < rank+1; n++)
		mag_size *= mag_dims[n];
	double *mag_decomposed;
	mag_decomposed = new double[2*mag_size];
	for(int n = 0; n < mag_size; n++){
		mag_decomposed[2*n+0] = real(data[n]);
		mag_decomposed[2*n+1] = imag(data[n]);
	}
	mag_dims[rank+1] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+2, mag_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(mag_decomposed, PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();
		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeMagnetization: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeMagnetization: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeMagnetization: While writing to " << name << "\n";
		exit(1);
	}

	delete [] mag_decomposed;
}

/*void FileWriter::writeLDOS(const double *ldos, int rank, const int *dims, double l_lim, double u_lim, int resolution, string name, string path){
	init();

	hsize_t ldos_dims[rank+1];//Last dimension is for energy
	for(int n = 0; n < rank; n++)
		ldos_dims[n] = dims[n];
	ldos_dims[rank] = resolution;

	double limits[2];
	limits[0] = u_lim;
	limits[1] = l_lim;
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+1, ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(ldos, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}
}*/

void FileWriter::writeLDOS(const Property::LDOS *ldos, string name, string path){
	init();

	int rank = ldos->getDimensions();
	const int *dims = ldos->getRanges();

	hsize_t ldos_dims[rank+1];//Last dimension is for energy
	for(int n = 0; n < rank; n++)
		ldos_dims[n] = dims[n];
	ldos_dims[rank] = ldos->getResolution();

	double limits[2];
	limits[0] = ldos->getUpperBound();
	limits[1] = ldos->getLowerBound();
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+1, ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(ldos->getData(), PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeLDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeLDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeLDOS: While writing to " << name << "\n";
		exit(1);
	}
}

/*void FileWriter::writeSP_LDOS(const complex<double> *sp_ldos, int rank, const int *dims, double l_lim, double u_lim, int resolution, string name, string path){
	init();

	const int NUM_MATRIX_ELEMENTS = 4;
	hsize_t sp_ldos_dims[rank+2];//Three last dimensions are for energy, spin components, and real/imaginary decomposition.
	for(int n = 0; n < rank; n++)
		sp_ldos_dims[n] = dims[n];
	sp_ldos_dims[rank] = resolution;
	sp_ldos_dims[rank+1] = NUM_MATRIX_ELEMENTS;

	int sp_ldos_size = 1;
	for(int n = 0; n < rank+2; n++)
		sp_ldos_size *= sp_ldos_dims[n];
	double *sp_ldos_decomposed;
	sp_ldos_decomposed = new double[2*sp_ldos_size];
	for(int n = 0; n < sp_ldos_size; n++){
		sp_ldos_decomposed[2*n+0] = real(sp_ldos[n]);
		sp_ldos_decomposed[2*n+1] = imag(sp_ldos[n]);
	}

	sp_ldos_dims[rank+2] = 2;

	double limits[2];
	limits[0] = u_lim;
	limits[1] = l_lim;
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;


	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+3, sp_ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(sp_ldos_decomposed, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
		dataspace.close();
	}
	catch(FileIException error){
		error.printError();
		return;
	}
	catch(DataSetIException error){
		error.printError();
		return;
	}
	catch(DataSpaceIException error){
		error.printError();
		return;
	}

	delete [] sp_ldos_decomposed;
}*/

void FileWriter::writeSpinPolarizedLDOS(const Property::SpinPolarizedLDOS *spinPolarizedLDOS, string name, string path){
	init();

	int rank = spinPolarizedLDOS->getDimensions();
	const int *dims = spinPolarizedLDOS->getRanges();
	const complex<double> *data = spinPolarizedLDOS->getData();

	const int NUM_MATRIX_ELEMENTS = 4;
	hsize_t sp_ldos_dims[rank+2];//Three last dimensions are for energy, spin components, and real/imaginary decomposition.
	for(int n = 0; n < rank; n++)
		sp_ldos_dims[n] = dims[n];
	sp_ldos_dims[rank] = spinPolarizedLDOS->getResolution();
	sp_ldos_dims[rank+1] = NUM_MATRIX_ELEMENTS;

	int sp_ldos_size = 1;
	for(int n = 0; n < rank+2; n++)
		sp_ldos_size *= sp_ldos_dims[n];
	double *sp_ldos_decomposed;
	sp_ldos_decomposed = new double[2*sp_ldos_size];
	for(int n = 0; n < sp_ldos_size; n++){
		sp_ldos_decomposed[2*n+0] = real(data[n]);
		sp_ldos_decomposed[2*n+1] = imag(data[n]);
	}

	sp_ldos_dims[rank+2] = 2;

	double limits[2];
	limits[0] = spinPolarizedLDOS->getUpperBound();
	limits[1] = spinPolarizedLDOS->getLowerBound();
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;


	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+3, sp_ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(sp_ldos_decomposed, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
		dataspace.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeSpinPolarizedLDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeSpinPolarizedLDOS: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeSpinPolarizedLDOS: While writing to " << name << "\n";
		exit(1);
	}

	delete [] sp_ldos_decomposed;
}

void FileWriter::write(const double *data, int rank, const int *dims, string name, string path){
	init();

	hsize_t data_dims[rank];
	for(int n = 0; n < rank; n++)
		data_dims[n] = dims[n];

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, data_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(data, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataset.close();
		file.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::write: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::write: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::write: While writing to " << name << "\n";
		exit(1);
	}
}

void FileWriter::writeAttributes(const int *attributes, const string *attribute_names, int num, string name, string path){
	init();

	const int ATTRIBUTES_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 1;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(ATTRIBUTES_RANK, limits_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::STD_I64BE, dataspace));
		for(int n = 0; n < num; n++){
			Attribute attribute = dataset.createAttribute(attribute_names[n], PredType::STD_I64BE, dataspace);
			attribute.write(PredType::NATIVE_INT, &(attributes[n]));
		}
		dataspace.close();
		dataset.close();

		file.close();
		dataspace.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
}

void FileWriter::writeAttributes(const double *attributes, const string *attribute_names, int num, string name, string path){
	init();

	const int ATTRIBUTES_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 1;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(ATTRIBUTES_RANK, limits_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		for(int n = 0; n < num; n++){
			Attribute attribute = dataset.createAttribute(attribute_names[n], PredType::IEEE_F64BE, dataspace);
			attribute.write(PredType::NATIVE_DOUBLE, &(attributes[n]));
		}
		dataspace.close();
		dataset.close();

		file.close();
		dataspace.close();
	}
	catch(FileIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileWriter::writeAttributes: While writing to " << name << "\n";
		exit(1);
	}
}

bool FileWriter::exists(){
	ifstream fin(filename);
	bool exists = fin.good();
	fin.close();

	return exists;
}

};	//End of namespace TBTK
