/** @file FileWriter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/FileWriter.h"
#include "../include/PropertyExtractor.h"
#include <string>
#include <sstream>
#include <iostream>
#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE
	using namespace H5;
#endif

using namespace std;

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

void FileWriter::writeAmplitudeSet(AmplitudeSet *amplitudeSet, string name, string path){
	init();

	int *asTable;
	int i_dims[2];
	amplitudeSet->tabulate(&asTable, i_dims);
//	PropertyExtractor::getTabulatedAmplitudeSet(&asTable, i_dims);

	const int RANK = 2;
	hsize_t dims[RANK];
	dims[0] = i_dims[0];
	dims[1] = i_dims[1];

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(RANK, dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::STD_I32BE, dataspace));
		dataset.write(asTable, PredType::NATIVE_INT);
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

	delete [] asTable;
}

void FileWriter::writeEV(double *ev, int size, string name, string path){
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
}

void FileWriter::writeDOS(double *dos, double u_lim, double l_lim, int resolution, string name, string path){
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
}

void FileWriter::writeDensity(double *density, int rank, int *dims, string name, string path){
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
}

/*void FileWriter::writeMAG(double *mag, int rank, int *dims, string name, string path){
	init();

	hsize_t mag_dims[rank+1];//Last dimension for spin components.
	for(int n = 0; n < rank; n++)
		mag_dims[n] = dims[n];
	const int NUM_SPIN_COMPONENTS = 3;
	mag_dims[rank] = NUM_SPIN_COMPONENTS;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank+1, mag_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(mag, PredType::NATIVE_DOUBLE);
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

void FileWriter::writeMAG(complex<double> *mag, int rank, int *dims, string name, string path){
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
}

void FileWriter::writeLDOS(double *ldos, int rank, int *dims, double u_lim, double l_lim, int resolution, string name, string path){
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
}

/*void FileWriter::writeSP_LDOS(double *sp_ldos, int rank, int *dims, double u_lim, double l_lim, int resolution, string name, string path){
	init();

	const int NUM_SPIN_COMPONENTS = 6;
	hsize_t sp_ldos_dims[rank+2];//Two last dimensions are for energy and spin components
	for(int n = 0; n < rank; n++)
		sp_ldos_dims[n] = dims[n];
	sp_ldos_dims[rank] = resolution;
	sp_ldos_dims[rank+1] = NUM_SPIN_COMPONENTS;

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

		DataSpace dataspace = DataSpace(rank+2, sp_ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(sp_ldos, PredType::NATIVE_DOUBLE);
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
}*/

void FileWriter::writeSP_LDOS(complex<double> *sp_ldos, int rank, int *dims, double u_lim, double l_lim, int resolution, string name, string path){
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
}

void FileWriter::write(double *data, int rank, int *dims, string name, string path){
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
}

void FileWriter::writeAttributes(int *attributes, string *attribute_names, int num, string name, string path){
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
}

void FileWriter::writeAttributes(double *attributes, string *attribute_names, int num, string name, string path){
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
}
