/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file FileWriter.cpp
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "../include/FileReader.h"
#include "../include/TBTKMacros.h"
#include "../include/Streams.h"

#include <string>
#include <sstream>
#include <H5Cpp.h>
#include <fstream>
#include <string>


#ifndef H5_NO_NAMESPACE
	using namespace H5;
#endif

using namespace std;

namespace TBTK{

bool FileReader::isInitialized = false;
string FileReader::filename = "TBTKResults.h5";

Model* FileReader::readModel(string name, string path){
	Model *model = new Model();

	stringstream ss;
	ss << name << "AmplitudeSet";

	delete model->amplitudeSet;
	model->amplitudeSet = readAmplitudeSet(ss.str());
	model->construct();

	ss.str("");
	ss << name << "Geometry";
	model->geometry = readGeometry(model, ss.str());

	const int NUM_DOUBLE_ATTRIBUTES = 2;
	ss.str("");
	ss << name << "DoubleAttributes";
	double doubleAttributes[NUM_DOUBLE_ATTRIBUTES];
	string doubleAttributeNames[NUM_DOUBLE_ATTRIBUTES] = {"Temperature", "ChemicalPotential"};
	readAttributes(doubleAttributes, doubleAttributeNames, NUM_DOUBLE_ATTRIBUTES, ss.str());

	model->setTemperature(doubleAttributes[0]);
	model->setChemicalPotential(doubleAttributes[1]);

	const int NUM_INT_ATTRIBUTES = 1;
	ss.str("");
	ss << name << "IntAttributes";
	int intAttributes[NUM_INT_ATTRIBUTES];
	string intAttributeNames[NUM_INT_ATTRIBUTES] = {"Statistics"};
	readAttributes(intAttributes, intAttributeNames, NUM_INT_ATTRIBUTES, ss.str());

	model->setStatistics(static_cast<Model::Statistics>(intAttributes[0]));

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
		TBTKAssert(
			typeClassI == H5T_INTEGER,
			"FileReader::readAmplitudeSet()",
			"Indices data type is not integer.",
			""
		);
		DataSpace dataspaceI = datasetI.getSpace();

		DataSet datasetA = file.openDataSet(ssA.str());
		H5T_class_t typeClassA = datasetA.getTypeClass();
		TBTKAssert(
			typeClassA == H5T_FLOAT,
			"FileReader::readAmplitudeSet()",
			"Amplitudes data type is not double.",
			""
		);
		DataSpace dataspaceA = datasetA.getSpace();

		hsize_t dims_internalI[3];
		dataspaceI.getSimpleExtentDims(dims_internalI, NULL);
		int numHoppingAmplitudes = dims_internalI[0];
		int maxIndexSize = dims_internalI[2];

		int *indices = new int[2*maxIndexSize*numHoppingAmplitudes];
		complex<double> *amplitudes = new complex<double>[numHoppingAmplitudes];

		datasetI.read(indices, PredType::NATIVE_INT, dataspaceI);
		datasetA.read(amplitudes, PredType::NATIVE_DOUBLE, dataspaceA);

		datasetI.close();
		dataspaceI.close();
		datasetA.close();
		dataspaceA.close();

		file.close();

		amplitudeSet = new AmplitudeSet();
		for(int n = 0; n < numHoppingAmplitudes; n++){
			vector<int> from;
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + c];
				if(i == -1)
					break;
				else
					from.push_back(i);
			}
			vector<int> to;
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + maxIndexSize + c];
				if(i == -1)
					break;
				else
					to.push_back(i);
			}

			amplitudeSet->addHA(HoppingAmplitude(amplitudes[n], to, from));
		}
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
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

		DataSet datasetC;
		try{
			 datasetC = file.openDataSet(ssC.str());
		}
		catch(...){
			return NULL;
		}
		H5T_class_t typeClassC = datasetC.getTypeClass();
		TBTKAssert(
			typeClassC == H5T_FLOAT,
			"FileReader::readGeometry()",
			"Coordinates data type is not double.",
			""
		);
		DataSpace dataspaceC = datasetC.getSpace();

		DataSet datasetS;
		try{
			datasetS = file.openDataSet(ssS.str());
		}
		catch(...){
			datasetC.close();
			dataspaceC.close();
			return NULL;
		}
		H5T_class_t typeClassS = datasetS.getTypeClass();
		TBTKAssert(
			typeClassS == H5T_INTEGER,
			"FileReader::readGeometry()",
			"Specifiers data type is not integer.",
			""
		);
		DataSpace dataspaceS = datasetS.getSpace();

		hsize_t dims_internalC[2];
		dataspaceC.getSimpleExtentDims(dims_internalC, NULL);
		int dimensions = dims_internalC[1];

		hsize_t dims_internalS[2];
		dataspaceS.getSimpleExtentDims(dims_internalS, NULL);
		int numSpecifiers = dims_internalS[1];

		geometry = new Geometry(dimensions, numSpecifiers, model);

		datasetC.read(geometry->coordinates, PredType::NATIVE_DOUBLE, dataspaceC);
		if(numSpecifiers != 0)
			datasetS.read(geometry->specifiers, PredType::NATIVE_INT, dataspaceS);

		datasetC.close();
		dataspaceC.close();
		datasetS.close();
		dataspaceS.close();

		file.close();
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return geometry;
}

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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::readEigenValues()",
			"Data type is not double.",
			""
		);

		DataSpace dataspace = dataset.getSpace();

		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		size = dims_internal[0];

		eigenValues = new Property::EigenValues(size);

		dataset.read(eigenValues->data, PredType::NATIVE_DOUBLE, dataspace);
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << "\n",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return eigenValues;
}

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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::readDOS()",
			"Data type is not double.",
			""
		);

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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return dos;
}

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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::readDensity()",
			"Data type is not double.",
			""
		);

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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ",",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return density;
}

Property::Magnetization* FileReader::readMagnetization(
	string name,
	string path
){
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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::readMAG()",
			"Data type is not double.",
			""
		);

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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return magnetization;
}

Property::LDOS* FileReader::readLDOS(string name, string path){
	TBTKExit(
		"FileReader::readLDOS()",
		"Not yet implemented.",
		""
	);

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

Property::SpinPolarizedLDOS* FileReader::readSpinPolarizedLDOS(
	string name,
	string path
){
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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::readSpinPolarizedLDOS()",
			"Data type is not double.",
			""
		);

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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return spinPolarizedLDOS;
}

void FileReader::read(
	double **data,
	int *rank,
	int **dims,
	string name,
	string path
){
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
		TBTKAssert(
			typeClass == H5T_FLOAT,
			"FileReader::read()",
			"Data type is not double.",
			""
		);

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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

void FileReader::readAttributes(
	int *attributes,
	string *attribute_names,
	int num,
	string name,
	string path
){
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
			TBTKAssert(
				type == PredType::STD_I64BE,
				"FileReader::readAttribues()",
				"The attribute '" << attribute_names[n] << "' is not of integer type.",
				""
			);
			attribute.read(PredType::NATIVE_INT, &(attributes[n]));
		}
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

void FileReader::readAttributes(
	double *attributes,
	string *attribute_names,
	int num,
	string name,
	string path
){
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
			TBTKAssert(
				type == PredType::IEEE_F64BE,
				"FileReader::readAttribues()",
				"The attribute '" << attribute_names[n] << "' is not of double type.",
				""
			);
			attribute.read(PredType::NATIVE_DOUBLE, &(attributes[n]));
		}
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

//herr_t
//attr_info(hid_t loc_id, const char *name, void *opdata)
//{
//    hid_t attr, atype, aspace;  /* Attribute, datatype, dataspace identifiers */
//    char    *string_out=NULL;
//    int   rank;
//    hsize_t sdim[64];
//    herr_t ret;
//    int i, j ;
//    size_t size, totsize;
//    size_t npoints;             /* Number of elements in the array attribute. */
//    int point_out;
//    float *float_array;         /* Pointer to the array attribute. */
//    H5S_class_t  classt;
//
//    /* avoid warnings */
//    opdata = opdata;
//
//    /*  Open the attribute using its name.  */
//    attr = H5Aopen_name(loc_id, name);
//
//    /*  Display attribute name.  */
//    printf("\nName : ");
//    puts(name);
//
//    /* Get attribute datatype, dataspace, rank, and dimensions.  */
//    atype  = H5Aget_type(attr);
//    aspace = H5Aget_space(attr);
//    rank = H5Sget_simple_extent_ndims(aspace);
//    ret = H5Sget_simple_extent_dims(aspace, sdim, NULL);
//
//    /* Get dataspace type */
//    classt = H5Sget_simple_extent_type (aspace);
//    printf ("H5Sget_simple_extent_type (aspace) returns: %i\n", classt);
//
//    /* Display rank and dimension sizes for the array attribute.  */
//    if(rank > 0) {
//       printf("Rank : %d \n", rank);
//       printf("Dimension sizes : ");
//       for (i=0; i< rank; i++) printf("%d ", (int)sdim[i]);
//       printf("\n");
//    }
//
//    if (H5T_INTEGER == H5Tget_class(atype)) {
//       printf("Type : INTEGER \n");
//       ret  = H5Aread(attr, atype, &point_out);
//       printf("The value of the attribute \"Integer attribute\" is %d \n",
//               point_out);
//    }
//
//    if (H5T_FLOAT == H5Tget_class(atype)) {
//       printf("Type : FLOAT \n");
//       npoints = H5Sget_simple_extent_npoints(aspace);
//       float_array = (float *)malloc(sizeof(float)*(int)npoints);
//       ret = H5Aread(attr, atype, float_array);
//       printf("Values : ");
//       for( i = 0; i < (int)npoints; i++) printf("%f ", float_array[i]);
//       printf("\n");
//       free(float_array);
//    }
//
//    if (H5T_STRING == H5Tget_class (atype)) {
//      printf ("Type: STRING \n");
//      size = H5Tget_size (atype);
//      printf ("Size of Each String is: %i\n", size);
//      totsize = size*sdim[0]*sdim[1];
//      string_out = reinterpret_cast<char*>(calloc (totsize, sizeof(char)));
//      ret = H5Aread(attr, atype, string_out);
//      printf("The value of the attribute with index 2 is:\n");
//      j=0;
//      for (i=0; i<totsize; i++) {
//        printf ("%c", string_out[i]);
//        if (j==3) {
//          printf(" ");
//          j=0;
//        }
//        else j++;
//      }
//      printf ("\n");
//    }
//
//    ret = H5Tclose(atype);
//    ret = H5Sclose(aspace);
//    ret = H5Aclose(attr);
//
//    return 0;
//}


Util::ParameterSet* FileReader::readParameterSet(
    string name,
	string path
){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();

        Util::ParameterSet *ps = new Util::ParameterSet();

//        unsigned int num = getNumAttributes(filename, name + "Int");

        H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet(name + "Int");

        unsigned int num = dataset.getNumAttrs();

		for(unsigned int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(n);

			DataType type = attribute.getDataType();
            string nameAttribute;
			nameAttribute = attribute.getName();

			TBTKAssert(
				type == PredType::STD_I64BE,
				"FileReader::readAttribues()",
				"The attribute '" << nameAttribute << "' is not of integer type.",
				""
			);
			int value;
			attribute.read(PredType::NATIVE_INT, &value);
			ps->addInt(nameAttribute, value);
		}

        dataset = file.openDataSet(name + "Double");
        num = dataset.getNumAttrs();

		for(unsigned int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(n);

			DataType type = attribute.getDataType();
            string nameAttribute;
			nameAttribute = attribute.getName();

			TBTKAssert(
				type == PredType::IEEE_F64BE,
				"FileReader::readAttribues()",
				"The attribute '" << nameAttribute << "' is not of double type.",
				""
			);
			double value;
			attribute.read(PredType::NATIVE_DOUBLE, &value);
			ps->addDouble(nameAttribute, value);
		}

        dataset = file.openDataSet(name + "Complex");
        num = dataset.getNumAttrs();
        complex<double> complexValue = 0;
        complex<double> realOne(1,0);
        complex<double> i(0,1);

		for(unsigned int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(n);

			DataType type = attribute.getDataType();
            string nameAttribute;
			nameAttribute = attribute.getName();

			TBTKAssert(
				type == PredType::IEEE_F64BE,
				"FileReader::readAttribues()",
				"The attribute '" << nameAttribute << "' is not of complex type.",
				""
			);
			double value;
			attribute.read(PredType::NATIVE_DOUBLE, &value);

            if(n%2)
            {
                complexValue += value*i;
                ps->addComplex(nameAttribute.erase(nameAttribute.size()-3), complexValue);
                complexValue = 0;
            }
            else
            {
                complexValue += value*realOne;
            }
		}

        dataset = file.openDataSet(name + "String");
        num = dataset.getNumAttrs();

		for(unsigned int n = 0; n < num; n++){
			Attribute attribute = dataset.openAttribute(n);

			DataType type = attribute.getDataType();
            string nameAttribute = attribute.getName();
			unsigned int memLength = attribute.getInMemDataSize();
			StrType strDataType(PredType::C_S1, memLength);

			TBTKAssert(
				type == strDataType,
				"FileReader::readParameterSet()",
				"The attribute '" << nameAttribute << "' is not of string type.",
				""
			);
			string value;
			attribute.read(type, value);
			ps->addString(nameAttribute, value);
		}
		return ps;
	}
	catch(FileIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

bool FileReader::exists(){
	ifstream fin(filename);
	bool exists = fin.good();
	fin.close();

	return exists;
}

};	//End of namespace TBTK
