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

Model* FileReader::readModel(string name, string path){
	Model *model = NULL;

	stringstream ss;
	ss << name << "AmplitudeSet";

//	readAmplitude();

	return model;
}

AmplitudeSet* FileReader::readAmplitudeSet(string name, string path){
	AmplitudeSet *amplitudeSet = NULL;

	try{
		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		stringstream ssI;
		ssI << path;
		if(path.back() != '/')
			ssI << "/";
		ssI << name << "Indices";

		stringstream ssA;
		ssA << path;
		if(path.back() != '/')
			ssA << "/";
		ssA << name << "Amplitudes";

		DataSet datasetI = file.openDataSet(ssI.str());
		H5T_class_t typeClassI = datasetI.getTypeClass();
		if(typeClassI != H5T_INTEGER){
			cout << "Error in FileReader::readAmplitudeSet: Indices data type is not integer.\n";
			exit(1);
		}
		DataSpace dataspaceI = datasetI.getSpace();

		DataSet datasetA = file.openDataSet(ssA.str());
		H5T_class_t typeClassA = datasetA.getTypeClass();
		if(typeClassA != H5T_FLOAT){
			cout << "Error in FileReader::readAmplitudeSet: Amplitudes data type is not double.\n";
			exit(1);
		}
		DataSpace dataspaceA = datasetA.getSpace();

		hsize_t dims_internalI[3];
		dataspaceI.getSimpleExtentDims(dims_internalI, NULL);
		int numHoppingAmplitudes = dims_internalI[0];
		int maxIndexSize = dims_internalI[2];

		int *indices = new int[2*maxIndexSize*numHoppingAmplitudes];
		complex<double> *amplitudes = new complex<double>[numHoppingAmplitudes];

		cout << "8\n";
		datasetI.read(indices, PredType::NATIVE_INT, dataspaceI);
		datasetA.read(amplitudes, PredType::NATIVE_DOUBLE, dataspaceA);

		cout << "9\n";
		datasetI.close();
		dataspaceI.close();
		datasetA.close();
		dataspaceA.close();

		file.close();

		amplitudeSet = new AmplitudeSet();
		for(int n = 0; n < numHoppingAmplitudes; n++){
			Index from({});
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + c];
				if(i == -1)
					break;
				else
					from.indices.push_back(i);
			}
			Index to({});
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + maxIndexSize + c];
				if(i == -1)
					break;
				else
					to.indices.push_back(i);
			}

			amplitudeSet->addHA(HoppingAmplitude(amplitudes[n], to, from));
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

	return amplitudeSet;
}

Geometry* FileReader::readGeometry(Model *model, string name, string path){
	Geometry *geometry = NULL;

	try{
		Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		stringstream ssC;
		ssC << path;
		if(path.back() != '/')
			ssC << "/";
		ssC << name << "Coordinates";

		stringstream ssS;
		ssS << path;
		if(path.back() != '/')
			ssS << "/";
		ssS << name << "Specifiers";

		DataSet datasetC = file.openDataSet(ssC.str());
		H5T_class_t typeClassC = datasetC.getTypeClass();
		if(typeClassC != H5T_FLOAT){
			cout << "Error in FileReader::readGeometry: Coordinates data type is not double.\n";
			exit(1);
		}
		DataSpace dataspaceC = datasetC.getSpace();

		DataSet datasetS = file.openDataSet(ssS.str());
		H5T_class_t typeClassS = datasetS.getTypeClass();
		if(typeClassS != H5T_INTEGER){
			cout << "Error in FileReader::readGeometry: Specifiers data type is not integer.\n";
			exit(1);
		}
		DataSpace dataspaceS = datasetS.getSpace();

		hsize_t dims_internalC[2];
		dataspaceC.getSimpleExtentDims(dims_internalC, NULL);
		int dimensions = dims_internalC[1];

		hsize_t dims_internalS[2];
		dataspaceS.getSimpleExtentDims(dims_internalS, NULL);
		int numSpecifiers = dims_internalS[1];

		geometry = new Geometry(dimensions, numSpecifiers, model);

		datasetC.read(geometry->coordinates, PredType::NATIVE_DOUBLE, dataspaceC);
		datasetS.read(geometry->specifiers, PredType::NATIVE_INT, dataspaceS);

		datasetC.close();
		dataspaceC.close();
		datasetS.close();
		dataspaceS.close();

		file.close();
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

	return geometry;
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
