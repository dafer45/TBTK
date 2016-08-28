/** @file FileWriter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/FileReader.h"
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

bool FileReader::isInitialized = false;
string FileReader::filename = "TBTKResults.h5";

void FileReader::readAmplitudeSet(AmplitudeSet **amplitudeSet, string name, string path){
	cout << "Error in FileReader::readAmplitudeSet: Not yet implemented.\n";
	exit(1);

/*	int *asTable;
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
		cout << "Error in FileReader::getAmplitudeSet(): While reading " << filename << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::getAmplitudeSet(): While reading " << filename << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::getAmplitudeSet(): While reading " << filename << "\n";
		exit(1);
	}

	delete [] asTable;*/
}

/*void FileReader::readEigenValues(double **ev, int *size, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readEigenValues: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		
		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*size = dims_internal[0];

		*ev = new double[*size];
		dataset.read(*ev, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}*/

Property::EigenValues* FileReader::readEigenValues(string name, string path){
	Property::EigenValues *eigenValues = NULL;
	int size;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readEigenValues: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		
		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		size = dims_internal[0];

		eigenValues = new Property::EigenValues(size);

		dataset.read(eigenValues->data, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}

	return eigenValues;
}

/*void FileReader::readDOS(double **dos, double *l_lim, double *u_lim, int *resolution, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readDOS: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		
		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*resolution = dims_internal[0];

		*dos = new double[*resolution];
		dataset.read(*dos, PredType::NATIVE_DOUBLE, dataspace);

		Attribute attribute = dataset.openAttribute("UpLowLimits");
		double limits[2];
		attribute.read(PredType::NATIVE_DOUBLE, limits);
		*u_lim = limits[0];
		*l_lim = limits[1];
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}*/

Property::DOS* FileReader::readDOS(string name, string path){
	Property::DOS *dos = NULL;
	double lowerBound;
	double upperBound;
	int resolution;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readDOS: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		
		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		resolution = dims_internal[0];

		Attribute attribute = dataset.openAttribute("UpLowLimits");
		double limits[2];
		attribute.read(PredType::NATIVE_DOUBLE, limits);
		upperBound = limits[0];
		lowerBound = limits[1];

		dos = new Property::DOS(lowerBound, upperBound, resolution);

		dataset.read(dos->data, PredType::NATIVE_DOUBLE, dataspace);

	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}

	return dos;
}

/*void FileReader::readDensity(double **density, int *rank, int **dims, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readDensity: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		*rank = dataspace.getSimpleExtentNdims();

		hsize_t *dims_internal = new hsize_t[*rank];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*dims = new int[*rank];
		for(int n = 0; n < *rank; n++)
			(*dims)[n] = dims_internal[n];

		delete [] dims_internal;

		int size = 1;
		for(int n = 0; n < *rank; n++)
			size *= (*dims)[n];

		*density = new double[size];
		dataset.read(*density, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}*/

Property::Density* FileReader::readDensity(string name, string path){
	Property::Density *density = NULL;
	int rank;
	int *dims;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readDensity: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		rank = dataspace.getSimpleExtentNdims();

		hsize_t *dims_internal = new hsize_t[rank];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		dims = new int[rank];
		for(int n = 0; n < rank; n++)
			dims[n] = dims_internal[n];
		delete [] dims_internal;

		density = new Property::Density(rank, dims);
		delete [] dims;

		dataset.read(density->data, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}

	return density;
}

/*void FileReader::readMAG(complex<double> **mag, int *rank, int **dims, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readMAG: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		int rank_internal = dataspace.getSimpleExtentNdims();
		*rank = rank_internal-2;//Last two dimensions are for matrix elements and real/imaginary decomposition.

		hsize_t *dims_internal = new hsize_t[rank_internal];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*dims = new int[*rank];
		for(int n = 0; n < rank_internal; n++)
			(*dims)[n] = dims_internal[n];

		int size = 1;
		for(int n = 0; n < rank_internal; n++)
			size *= dims_internal[n];

		double *mag_internal = new double[size];
		*mag = new complex<double>[size/2];
		dataset.read(mag_internal, PredType::NATIVE_DOUBLE, dataspace);
		for(int n = 0; n < size/2; n++)
			(*mag)[n] = complex<double>(mag_internal[2*n+0], mag_internal[2*n+1]);

		delete [] mag_internal;
		delete [] dims_internal;
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}*/

Property::Magnetization* FileReader::readMagnetization(string name, string path){
	Property::Magnetization *magnetization = NULL;
	int rank;
	int *dims;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readMAG: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		int rank_internal = dataspace.getSimpleExtentNdims();
		rank = rank_internal-2;//Last two dimensions are for matrix elements and real/imaginary decomposition.

		hsize_t *dims_internal = new hsize_t[rank_internal];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		dims = new int[rank];
		for(int n = 0; n < rank; n++)
			dims[n] = dims_internal[n];

		magnetization = new Property::Magnetization(rank, dims);
		delete [] dims;

		int size = 1;
		for(int n = 0; n < rank_internal; n++)
			size *= dims_internal[n];

		double *mag_internal = new double[size];
		dataset.read(mag_internal, PredType::NATIVE_DOUBLE, dataspace);
		for(int n = 0; n < size/2; n++)
			magnetization->data[n] = complex<double>(mag_internal[2*n+0], mag_internal[2*n+1]);

		delete [] mag_internal;
		delete [] dims_internal;
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}

	return magnetization;
}

Property::LDOS* FileReader::readLDOS(string name, string path){
	cout << "Error in FileReader::readLDOS: Not yet implemented.\n";
	exit(1);

/*	hsize_t ldos_dims[rank+1];//Last dimension is for energy
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
	}*/
}

/*void FileReader::readSP_LDOS(complex<double> **sp_ldos, int *rank, int **dims, double *l_lim, double *u_lim, int *resolution, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readSP_LDOS: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		int rank_internal = dataspace.getSimpleExtentNdims();
		*rank = rank_internal-3;//Three last dimensions are for energy, spin components, and real/imaginary decomposition.

		hsize_t *dims_internal = new hsize_t[rank_internal];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*dims = new int[*rank];
		for(int n = 0; n < rank_internal; n++)
			(*dims)[n] = dims_internal[n];

		int size = 1;
		for(int n = 0; n < rank_internal; n++)
			size *= dims_internal[n];

		double *sp_ldos_internal = new double[size];
		*sp_ldos = new complex<double>[size/2];
		dataset.read(sp_ldos_internal, PredType::NATIVE_DOUBLE, dataspace);
		for(int n = 0; n < size/2; n++)
			(*sp_ldos)[n] = complex<double>(sp_ldos_internal[2*n+0], sp_ldos_internal[2*n+1]);

		*resolution = dims_internal[rank_internal-3];

		delete [] sp_ldos_internal;
		delete [] dims_internal;

		Attribute attribute = dataset.openAttribute("UpLowLimits");
		double limits[2];
		attribute.read(PredType::NATIVE_DOUBLE, limits);
		*u_lim = limits[0];
		*l_lim = limits[1];
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}*/

Property::SpinPolarizedLDOS* FileReader::readSpinPolarizedLDOS(string name, string path){
	Property::SpinPolarizedLDOS *spinPolarizedLDOS = NULL;
	int rank;
	int *dims;
	double lowerBound;
	double upperBound;
	int resolution;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::readSP_LDOS: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		int rank_internal = dataspace.getSimpleExtentNdims();
		rank = rank_internal-3;//Three last dimensions are for energy, spin components, and real/imaginary decomposition.

		hsize_t *dims_internal = new hsize_t[rank_internal];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		dims = new int[rank];
		for(int n = 0; n < rank; n++)
			dims[n] = dims_internal[n];
		resolution = dims_internal[rank];

		Attribute attribute = dataset.openAttribute("UpLowLimits");
		double limits[2];
		attribute.read(PredType::NATIVE_DOUBLE, limits);
		upperBound = limits[0];
		lowerBound = limits[1];

		spinPolarizedLDOS = new Property::SpinPolarizedLDOS(rank, dims, lowerBound, upperBound, resolution);
		delete [] dims;

		int size = 1;
		for(int n = 0; n < rank_internal; n++)
			size *= dims_internal[n];

		double *sp_ldos_internal = new double[size];
		dataset.read(sp_ldos_internal, PredType::NATIVE_DOUBLE, dataspace);
		for(int n = 0; n < size/2; n++)
			spinPolarizedLDOS->data[n] = complex<double>(sp_ldos_internal[2*n+0], sp_ldos_internal[2*n+1]);

		delete [] sp_ldos_internal;
		delete [] dims_internal;
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}

	return spinPolarizedLDOS;
}

void FileReader::read(double **data, int *rank, int **dims, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		if(typeClass != H5T_FLOAT){
			cout << "Error in FileReader::read: Data type is not double.\n";
			exit(1);
		}

		DataSpace dataspace = dataset.getSpace();
		*rank = dataspace.getSimpleExtentNdims();

		hsize_t *dims_internal = new hsize_t[*rank];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		*dims = new int[*rank];
		for(int n = 0; n < *rank; n++)
			(*dims)[n] = dims_internal[n];
		delete [] dims_internal;

		int size = 1;
		for(int n = 0; n < *rank; n++)
			size *= (*dims)[n];

		*data = new double[size];
		dataset.read(*data, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}

void FileReader::readAttributes(int *attributes, string *attribute_names, int num, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		DataSpace dataspace = dataset.getSpace();

		for(int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(attribute_names[n]);
			DataType type = attribute.getDataType();
			if(!(type == PredType::STD_I64BE)){
				cout << "Error in FileReader::readAttribues: The attribute '" << attribute_names[n] << "' is not of integer type.\n";
				exit(1);
			}
			attribute.read(PredType::NATIVE_INT, &(attributes[n]));
		}
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}

void FileReader::readAttributes(double *attributes, string *attribute_names, int num, string name, string path){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		DataSpace dataspace = dataset.getSpace();

		for(int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(attribute_names[n]);
			DataType type = attribute.getDataType();
			if(!(type == PredType::IEEE_F64BE)){
				cout << "Error in FileReader::readAttribues: The attribute '" << attribute_names[n] << "' is not of double type.\n";
				exit(1);
			}
			attribute.read(PredType::NATIVE_DOUBLE, &(attributes[n]));
		}
	}
	catch(FileIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSetIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
	catch(DataSpaceIException error){
		cout << "Error in FileReader::read: While reading " << name << "\n";
		exit(1);
	}
}

bool FileReader::exists(){
	ifstream fin(filename);
	bool exists = fin.good();
	fin.close();

	return exists;
}

};	//End of namespace TBTK
