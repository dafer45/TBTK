#include "../include/FileWriter.h"
#include "../include/FileIO.h"
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
string FileWriter::filename = "TBResults.h5";

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

void FileWriter::writeLDOS(double *ldos, int rank, int *dims, string name, string path){
	init();

	hsize_t ldos_dims[rank];
	for(int n = 0; n < rank; n++)
		ldos_dims[n] = dims[n];

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, ldos_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(ldos, PredType::NATIVE_DOUBLE);
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

void FileWriter::writeMAG(double *mag, int rank, int *dims, string name, string path){
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
}

void FileWriter::writeSP_LDOS_E(double *sp_ldos_e, int rank, int *dims, double u_lim, double l_lim, int resolution, string name, string path){
	init();

	const int NUM_SPIN_COMPONENTS = 6;
	hsize_t sp_ldos_e_dims[rank+2];//Two last dimensions are for energy and spin components
	for(int n = 0; n < rank; n++)
		sp_ldos_e_dims[n] = dims[n];
	sp_ldos_e_dims[rank] = resolution;
	sp_ldos_e_dims[rank+1] = NUM_SPIN_COMPONENTS;

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

		DataSpace dataspace = DataSpace(rank+2, sp_ldos_e_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(sp_ldos_e, PredType::NATIVE_DOUBLE);
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
