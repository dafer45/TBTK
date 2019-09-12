/* Copyright 2016 Kristofer Björnson and Andreas Theiler
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

#include "TBTK/FileReader.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <H5Cpp.h>

#include <fstream>
#include <sstream>
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
	ss << name << "HoppingAmplitudeSet";

	HoppingAmplitudeSet *hoppingAmplitudeSet = readHoppingAmplitudeSet(ss.str());
	model->singleParticleContext.hoppingAmplitudeSet = *hoppingAmplitudeSet;
	delete hoppingAmplitudeSet;
	model->construct();

	ss.str("");
	ss << name << "Geometry";
	model->singleParticleContext.geometry = *readGeometry(model, ss.str());

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

	model->setStatistics(static_cast<Statistics>(intAttributes[0]));

	return model;
}

HoppingAmplitudeSet* FileReader::readHoppingAmplitudeSet(
	string name,
	string path
){
	HoppingAmplitudeSet *hoppingAmplitudeSet = NULL;

	try{
		H5::Exception::dontPrint();
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
			"FileReader::readHoppingAmplitudeSet()",
			"Indices data type is not integer.",
			""
		);
		DataSpace dataspaceI = datasetI.getSpace();

		DataSet datasetA = file.openDataSet(ssA.str());
		H5T_class_t typeClassA = datasetA.getTypeClass();
		TBTKAssert(
			typeClassA == H5T_FLOAT,
			"FileReader::readHoppingAmplitudeSet()",
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

		hoppingAmplitudeSet = new HoppingAmplitudeSet();
		for(int n = 0; n < numHoppingAmplitudes; n++){
			vector<Subindex> from;
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + c];
				if(i == -1)
					break;
				else
					from.push_back(i);
			}
			vector<Subindex> to;
			for(int c = 0; c < maxIndexSize; c++){
				int i = indices[2*maxIndexSize*n + maxIndexSize + c];
				if(i == -1)
					break;
				else
					to.push_back(i);
			}

			hoppingAmplitudeSet->add(
				HoppingAmplitude(amplitudes[n], to, from)
			);
		}
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return hoppingAmplitudeSet;
}

Geometry* FileReader::readGeometry(Model *model, string name, string path){
	//Se ATTENTION comment below.
	TBTKExit(
		"FileReader::readGeometry()",
		"This function is no longer supported.",
		"The FileReader and FileWriter are deprecated. Please use"
		<< " serialization instead."
	);

	Geometry *geometry = NULL;

	try{
		H5::Exception::dontPrint();
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

		//ATTENTION: The coordinate data can no longer be easily stored
		//by the FileWriter. The following lines are therefore not
		//valid any longer and commented out and left here in case they
		//are needed for future reference.
		geometry = nullptr;
/*		hsize_t dims_internalC[2];
		dataspaceC.getSimpleExtentDims(dims_internalC, NULL);
		int dimensions = dims_internalC[1];

		hsize_t dims_internalS[2];
		dataspaceS.getSimpleExtentDims(dims_internalS, NULL);
		int numSpecifiers = dims_internalS[1];

		geometry = new Geometry(dimensions, numSpecifiers, &model->getHoppingAmplitudeSet());

		datasetC.read(geometry->coordinates, PredType::NATIVE_DOUBLE, dataspaceC);
		if(numSpecifiers != 0)
			datasetS.read(geometry->specifiers, PredType::NATIVE_INT, dataspaceS);*/

		datasetC.close();
		dataspaceC.close();
		datasetS.close();
		dataspaceS.close();

		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return geometry;
}

IndexTree* FileReader::readIndexTree(string name, string path){
	IndexTree *indexTree = NULL;

	try{
		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		DataSet dataset = file.openDataSet(ss.str());
		H5T_class_t typeClass = dataset.getTypeClass();
		TBTKAssert(
			typeClass == H5T_INTEGER,
			"FileReader::readIndexTree()",
			"Indices data type is not integer.",
			""
		);
		DataSpace dataspace = dataset.getSpace();

		hsize_t dims_internal[1];
		dataspace.getSimpleExtentDims(dims_internal, NULL);
		unsigned int size = dims_internal[0];

		int *serializedIndices = new int[size];

		dataset.read(serializedIndices, PredType::NATIVE_INT, dataspace);

		dataset.close();
		dataspace.close();

		file.close();

		vector<Index> indices;
		int subindicesLeft = 0;
		for(unsigned int n = 0; n < size; n++){
			if(subindicesLeft == 0){
//				indices.push_back(Index({}));
				indices.push_back(Index());
				subindicesLeft = serializedIndices[n];
			}
			else{
				subindicesLeft--;
				indices.back().pushBack(serializedIndices[n]);
			}
		}

		delete [] serializedIndices;

		indexTree = new IndexTree();
		for(unsigned int n = 0; n < indices.size(); n++)
			indexTree->add(indices.at(n));

		indexTree->generateLinearMap();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::readIndexTree()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::readIndexTree()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::readIndexTree()",
			"While reading " << name << ".",
			""
		);
	}

	return indexTree;
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

		H5::Exception::dontPrint();
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

		dataset.read(
			eigenValues->getDataRW().data(),
			PredType::NATIVE_DOUBLE,
			dataspace
		);
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << "\n",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}

	return eigenValues;
}

Property::WaveFunctions* FileReader::readWaveFunctions(string name, string path){
	Property::WaveFunctions *waveFunctions = NULL;

	int attributes[2];
	string attributeNames[2];
	attributeNames[0] = "Format";
	attributeNames[1] = "NumStates";
	stringstream ss;
	ss << name << "Attributes";
	readAttributes(
		attributes,
		attributeNames,
		2,
		ss.str(),
		path
	);

	IndexDescriptor::Format format = static_cast<IndexDescriptor::Format>(
		attributes[0]
	);

	switch(format){
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		IndexTree *indexTree = readIndexTree(
			ss.str(),
			path
		);

		ss.str("");
		ss << name << "States";
		int *serializedStates;
		int statesRank;
		int *statesDims;
		read(&serializedStates, &statesRank, &statesDims, ss.str(), path);
		TBTKAssert(
			statesRank == 1,
			"FileReader::readWaveFunctions()",
			"Unable to read 'states'.",
			"This should never happen, something is wrong with the input data."
		);
		vector<unsigned int> states;
		for(int n = 0; n < statesDims[0]; n++)
			states.push_back(serializedStates[n]);
		delete [] serializedStates;
		delete [] statesDims;

		complex<double> *data;
		int rank;
		int *dims;
		read(&data, &rank, &dims, name, path);

		waveFunctions = new Property::WaveFunctions(
			*indexTree,
			states,
			data
		);

		delete indexTree;
		delete [] data;
		delete [] dims;

		break;
	}
	default:
		TBTKExit(
			"FileReader::readWaveFunctions()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}

	return waveFunctions;
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

		H5::Exception::dontPrint();
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

		dataset.read(
			dos->getDataRW().data(),
			PredType::NATIVE_DOUBLE,
			dataspace
		);
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
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

	int attributes[1];
	string attributeNames[1];
	attributeNames[0] = "Format";
	stringstream ss;
	ss << name << "Attributes";
	readAttributes(
		attributes,
		attributeNames,
		1,
		ss.str(),
		path
	);

	IndexDescriptor::Format format = static_cast<IndexDescriptor::Format>(
		attributes[0]
	);

	switch(format){
	case IndexDescriptor::Format::Ranges:
	{
		int rank;
		int *dims;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
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

//			density = new Property::Density(rank, dims);
			density = new Property::Density(
				vector<int>(dims, dims + rank)
			);
			delete [] dims;

			dataset.read(
				density->getDataRW().data(),
				PredType::NATIVE_DOUBLE,
				dataspace
			);
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ",",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		IndexTree *indexTree = readIndexTree(
			ss.str(),
			path
		);

		int rank;
		int *dims;
		double *data;
		read(&data, &rank, &dims, name, path);

		density = new Property::Density(*indexTree, data);

		delete indexTree;
		delete [] dims;
		delete [] data;

		break;
	}
	default:
		TBTKExit(
			"FileReader::readDensity()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}

	return density;
}

Property::Magnetization* FileReader::readMagnetization(
	string name,
	string path
){
	Property::Magnetization *magnetization = NULL;

	int attributes[1];
	string attributeNames[1];
	attributeNames[0] = "Format";
	stringstream ss;
	ss << name << "Attributes";
	readAttributes(
		attributes,
		attributeNames,
		1,
		ss.str(),
		path
	);

	IndexDescriptor::Format format = static_cast<IndexDescriptor::Format>(
		attributes[0]
	);

	switch(format){
	case IndexDescriptor::Format::Ranges:
	{
		int rank;
		int *dims;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
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

//			magnetization = new Property::Magnetization(rank, dims);
			magnetization = new Property::Magnetization(
				vector<int>(dims, dims + rank)
			);
			delete [] dims;

			int size = 1;
			for(int n = 0; n < rank_internal; n++)
				size *= dims_internal[n];

			double *mag_internal = new double[size];
			dataset.read(
				mag_internal,
				PredType::NATIVE_DOUBLE,
				dataspace
			);
			std::vector<SpinMatrix> &data
				= magnetization->getDataRW();
			for(int n = 0; n < size/8; n++){
				data[n].at(0, 0) = complex<double>(mag_internal[8*n+0], mag_internal[8*n+1]);
				data[n].at(0, 1) = complex<double>(mag_internal[8*n+2], mag_internal[8*n+3]);
				data[n].at(1, 0) = complex<double>(mag_internal[8*n+4], mag_internal[8*n+5]);
				data[n].at(1, 1) = complex<double>(mag_internal[8*n+6], mag_internal[8*n+7]);
			}

			delete [] mag_internal;
			delete [] dims_internal;
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		IndexTree *indexTree = readIndexTree(
			ss.str(),
			path
		);

		int rank;
		int *dims;
		complex<double> *data_internal;
		read(&data_internal, &rank, &dims, name, path);

		SpinMatrix *data = new SpinMatrix[dims[0]/4];
		for(int n = 0; n < dims[0]/4; n++){
			data[n].at(0, 0) = data_internal[4*n + 0];
			data[n].at(0, 1) = data_internal[4*n + 1];
			data[n].at(1, 0) = data_internal[4*n + 2];
			data[n].at(1, 1) = data_internal[4*n + 3];
		}

		magnetization = new Property::Magnetization(*indexTree, data);

		delete indexTree;
		delete [] dims;
		delete [] data_internal;
		delete [] data;

		break;
	}
	default:
		TBTKExit(
			"FileReader::readMagnetization()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}

	return magnetization;
}

Property::LDOS* FileReader::readLDOS(string name, string path){
	Property::LDOS *ldos;

	int intAttributes[2];
	string intAttributeNames[2];
	intAttributeNames[0] = "Format";
	intAttributeNames[1] = "Resolution";
	stringstream ss;
	ss << name << "IntAttributes";
	readAttributes(
		intAttributes,
		intAttributeNames,
		2,
		ss.str(),
		path
	);

	double doubleAttributes[2];
	string doubleAttributeNames[2];
	doubleAttributeNames[0] = "LowerBound";
	doubleAttributeNames[1] = "UpperBound";
	ss.str("");
	ss << name << "DoubleAttributes";
	readAttributes(
		doubleAttributes,
		doubleAttributeNames,
		2,
		ss.str(),
		path
	);

	IndexDescriptor::Format format = static_cast<IndexDescriptor::Format>(
		intAttributes[0]
	);

	switch(format){
	case IndexDescriptor::Format::Ranges:
	{
		int rank;
		int *dims;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
			H5File file(filename, H5F_ACC_RDONLY);

			DataSet dataset = file.openDataSet(name);
			H5T_class_t typeClass = dataset.getTypeClass();
			TBTKAssert(
				typeClass == H5T_FLOAT,
				"FileReader::readLDOS()",
				"Data type is not double.",
				""
			);

			DataSpace dataspace = dataset.getSpace();
			int rank_internal = dataspace.getSimpleExtentNdims();
			rank = rank_internal-1;//Last dimension is for energy.

			hsize_t *dims_internal = new hsize_t[rank_internal];
			dataspace.getSimpleExtentDims(dims_internal, NULL);
			dims = new int[rank];
			for(int n = 0; n < rank; n++)
				dims[n] = dims_internal[n];

			ldos = new Property::LDOS(
				vector<int>(dims, dims + rank),
				doubleAttributes[0],
				doubleAttributes[1],
				intAttributes[1]
			);
			delete [] dims;

			int size = 1;
			for(int n = 0; n < rank_internal; n++)
				size *= dims_internal[n];

			std::vector<double> &ldos_data = ldos->getDataRW();
			dataset.read(
				ldos_data.data(),
				PredType::NATIVE_DOUBLE,
				dataspace
			);

			delete [] dims_internal;
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::readLDOS()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::readLDOS()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::readLDOS()",
				"While reading " << name << ".",
				""
			);
		}

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		IndexTree *indexTree = readIndexTree(
			ss.str(),
			path
		);

		int rank;
		int *dims;
		double *data;
		read(&data, &rank, &dims, name, path);

		ldos = new Property::LDOS(
			*indexTree,
			doubleAttributes[0],
			doubleAttributes[1],
			intAttributes[1],
			data
		);

		delete indexTree;
		delete [] dims;
		delete [] data;

		break;
	}
	default:
		TBTKExit(
			"FileReader::readLDOS()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}

	return ldos;
}

Property::SpinPolarizedLDOS* FileReader::readSpinPolarizedLDOS(
	string name,
	string path
){
	Property::SpinPolarizedLDOS *spinPolarizedLDOS = NULL;

	int intAttributes[2];
	string intAttributeNames[2];
	intAttributeNames[0] = "Format";
	intAttributeNames[1] = "Resolution";
	stringstream ss;
	ss << name << "IntAttributes";
	readAttributes(
		intAttributes,
		intAttributeNames,
		2,
		ss.str(),
		path
	);

	double doubleAttributes[2];
	string doubleAttributeNames[2];
	doubleAttributeNames[0] = "LowerBound";
	doubleAttributeNames[1] = "UpperBound";
	ss.str("");
	ss << name << "DoubleAttributes";
	readAttributes(
		doubleAttributes,
		doubleAttributeNames,
		2,
		ss.str(),
		path
	);

	IndexDescriptor::Format format = static_cast<IndexDescriptor::Format>(
		intAttributes[0]
	);

	switch(format){
	case IndexDescriptor::Format::Ranges:
	{
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

			H5::Exception::dontPrint();
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

//			spinPolarizedLDOS = new Property::SpinPolarizedLDOS(rank, dims, lowerBound, upperBound, resolution);
			spinPolarizedLDOS = new Property::SpinPolarizedLDOS(
				vector<int>(dims, dims + rank),
				lowerBound,
				upperBound,
				resolution
			);
			delete [] dims;

			int size = 1;
			for(int n = 0; n < rank_internal; n++)
				size *= dims_internal[n];

			double *sp_ldos_internal = new double[size];
			dataset.read(
				sp_ldos_internal,
				PredType::NATIVE_DOUBLE,
				dataspace
			);
			std::vector<SpinMatrix> &data
				= spinPolarizedLDOS->getDataRW();
			for(int n = 0; n < size/8; n++){
				data[n].at(0, 0) = complex<double>(sp_ldos_internal[8*n+0], sp_ldos_internal[8*n+1]);
				data[n].at(0, 1) = complex<double>(sp_ldos_internal[8*n+2], sp_ldos_internal[8*n+3]);
				data[n].at(1, 0) = complex<double>(sp_ldos_internal[8*n+4], sp_ldos_internal[8*n+5]);
				data[n].at(1, 1) = complex<double>(sp_ldos_internal[8*n+6], sp_ldos_internal[8*n+7]);
			}

			delete [] sp_ldos_internal;
			delete [] dims_internal;
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileReader::read()",
				"While reading " << name << ".",
				""
			);
		}

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		IndexTree *indexTree = readIndexTree(
			ss.str(),
			path
		);

		int rank;
		int *dims;
		complex<double> *data_internal;
		read(&data_internal, &rank, &dims, name, path);

		SpinMatrix *data = new SpinMatrix[dims[0]/4];
		for(int n = 0; n < dims[0]/4; n++){
			data[n].at(0, 0) = data_internal[4*n + 0];
			data[n].at(0, 1) = data_internal[4*n + 1];
			data[n].at(1, 0) = data_internal[4*n + 2];
			data[n].at(1, 1) = data_internal[4*n + 3];
		}

		spinPolarizedLDOS = new Property::SpinPolarizedLDOS(
			*indexTree,
			doubleAttributes[0],
			doubleAttributes[1],
			intAttributes[1],
			data
		);

		delete indexTree;
		delete [] dims;
		delete [] data_internal;
		delete [] data;

		break;
	}
	default:
		TBTKExit(
			"FileReader::readSpinPolarizedLDOS()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}

	return spinPolarizedLDOS;
}

void FileReader::read(
	int **data,
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

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDONLY);

		DataSet dataset = file.openDataSet(name);
		H5T_class_t typeClass = dataset.getTypeClass();
		TBTKAssert(
			typeClass == H5T_INTEGER,
			"FileReader::read()",
			"Data type is not int.",
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

		*data = new int[size];
		dataset.read(*data, PredType::NATIVE_INT, dataspace);
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
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

		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

void FileReader::read(
	complex<double> **data,
	int *rank,
	int **dims,
	string name,
	string path
){
	double *realData;
	double *imagData;
	int realRank;
	int imagRank;
	int *realDims;
	int *imagDims;

	stringstream ss;
	ss << name << "Real";
	read(&realData, &realRank, &realDims, ss.str(), path);
	ss.str("");
	ss << name << "Imag";
	read(&imagData, &imagRank, &imagDims, ss.str(), path);

	TBTKAssert(
		realRank == imagRank,
		"FileReader::read()",
		"While reading " << name << ": Incompatible ranks for real and imaginary data.",
		""
	);
	for(int n = 0; n < realRank; n++){
		TBTKAssert(
			realDims[n] == imagDims[n],
			"FileReader::read()",
			"While reading " << name << ": Incompatible dimensions for real and imaginary data.",
			""
		);
	}

	*rank = realRank;

	*dims = new int[*rank];
	for(int n = 0; n < *rank; n++)
		(*dims)[n] = realDims[n];

	unsigned int size = 1;
	for(int n = 0; n < *rank; n++)
		size *= (*dims)[n];

	*data = new complex<double>[size];
	for(unsigned int n = 0; n < size; n++)
		(*data)[n] = realData[n] + complex<double>(0, 1.)*imagData[n];

	delete [] realData;
	delete [] imagData;
	delete [] realDims;
	delete [] imagDims;
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

		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
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

		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
}

ParameterSet* FileReader::readParameterSet(
	string name,
	string path
){
	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		H5::Exception::dontPrint();
		ParameterSet *ps = new ParameterSet();

		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet(name + "Int");
		unsigned int numAttributes = dataset.getNumAttrs();

		for(unsigned int n = 0; n < numAttributes; n++){
			Attribute attribute = dataset.openAttribute(n);
			DataType type = attribute.getDataType();
			string attributeName;
			attributeName = attribute.getName();

			TBTKAssert(
				type == PredType::STD_I64BE,
				"FileReader::readParameterSet()",
				"The attribute '" << attributeName << "' is not of integer type.",
				""
			);
			int value;
			attribute.read(PredType::NATIVE_INT, &value);
			ps->addInt(attributeName, value);
		}

		dataset = file.openDataSet(name + "Double");
		numAttributes = dataset.getNumAttrs();

		for(unsigned int n = 0; n < numAttributes; n++){
			Attribute attribute = dataset.openAttribute(n);
			DataType type = attribute.getDataType();
			string attributeName;
			attributeName = attribute.getName();

			TBTKAssert(
				type == PredType::IEEE_F64BE,
				"FileReader::readParameterSet()",
				"The attribute '" << attributeName << "' is not of double type.",
				""
			);
			double value;
			attribute.read(PredType::NATIVE_DOUBLE, &value);
			ps->addDouble(attributeName, value);
		}

		dataset = file.openDataSet(name + "Complex");
		numAttributes = dataset.getNumAttrs();
		const complex<double> i(0,1);

		for(unsigned int n = 0; n < numAttributes; n++){
			Attribute attribute = dataset.openAttribute(n);
			DataType type = attribute.getDataType();
			string attributeName;
			attributeName = attribute.getName();
			const int COMPLEX_RANK = 1;
			const hsize_t complex_dims[COMPLEX_RANK] = {2};
			ArrayType complexDataType(PredType::NATIVE_DOUBLE, COMPLEX_RANK, complex_dims);

			TBTKAssert(
				type == complexDataType,
				"FileReader::readParameterSet()",
				"The attribute '" << attributeName << "' is not of complex type.",
				""
			);
			double value[2];
			attribute.read(complexDataType, value);
			complex<double> complexValue = value[0];
			complexValue += value[1]*i;
			ps->addComplex(attributeName, complexValue);
		}

		dataset = file.openDataSet(name + "String");
		numAttributes = dataset.getNumAttrs();

		for(unsigned int n = 0; n < numAttributes; n++){
			Attribute attribute = dataset.openAttribute(n);
			DataType type = attribute.getDataType();
			string attributeName = attribute.getName();
			unsigned int memLength = attribute.getInMemDataSize();
			StrType strDataType(PredType::C_S1, memLength);

			TBTKAssert(
				type == strDataType,
				"FileReader::readParameterSet()",
				"The attribute '" << attributeName << "' is not of string type.",
				""
			);
			string value;
			attribute.read(type, value);
			ps->addString(attributeName, value);
		}

		dataset = file.openDataSet(name + "Bool");
		numAttributes = dataset.getNumAttrs();

		for(unsigned int n = 0; n < numAttributes; n++){
			Attribute attribute = dataset.openAttribute(n);
			DataType type = attribute.getDataType();
			string attributeName;
			attributeName = attribute.getName();

			TBTKAssert(
				type == PredType::STD_I64BE,
				"FileReader::readParameterSet()",
				"The attribute '" << attributeName << "' is not of bool type.",
				""
			);
			int value;
			attribute.read(PredType::NATIVE_INT, &value);
			ps->addBool(attributeName, value);
		}
		return ps;
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileReader::read()",
			"While reading " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
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
