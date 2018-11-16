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

#include "TBTK/FileWriter.h"

#include <H5Cpp.h>

#include <fstream>
#include <sstream>
#include <string>

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
		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);
		file.close();
	}
	catch(FileIException &error){
		H5File file(filename, H5F_ACC_EXCL);
		file.close();
	}

	isInitialized = true;
}

void FileWriter::writeModel(const Model &model, string name, string path){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeModel()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	stringstream ss;
	ss << name << "HoppingAmplitudeSet";

	writeHoppingAmplitudeSet(model.getHoppingAmplitudeSet(), ss.str(), path);

	ss.str("");
	ss << name << "Geometry";
/*	if(model.getGeometry() != NULL)
		writeGeometry(*model.getGeometry(), ss.str(), path);*/
		writeGeometry(model.getGeometry(), ss.str(), path);


	const int NUM_DOUBLE_ATTRIBUTES = 2;
	ss.str("");
	ss << name << "DoubleAttributes";
	double doubleAttributes[NUM_DOUBLE_ATTRIBUTES] = {model.getTemperature(), model.getChemicalPotential()};
	string doubleAttributeNames[NUM_DOUBLE_ATTRIBUTES] = {"Temperature", "ChemicalPotential"};
	writeAttributes(doubleAttributes, doubleAttributeNames, NUM_DOUBLE_ATTRIBUTES, ss.str());

	const int NUM_INT_ATTRIBUTES = 1;
	ss.str("");
	ss << name << "IntAttributes";
	int intAttributes[NUM_INT_ATTRIBUTES] = {static_cast<int>(model.getStatistics())};
	string intAttributeNames[NUM_INT_ATTRIBUTES] = {"Statistics"};
	writeAttributes(intAttributes, intAttributeNames, NUM_INT_ATTRIBUTES, ss.str());
}

void FileWriter::writeHoppingAmplitudeSet(
	const HoppingAmplitudeSet &hoppingAmplitudeSet,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeHoppingAmplitudeSet()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	complex<double> *amplitudes;
	int *indices;
	int numHoppingAmplitudes;
	int maxIndexSize;
	hoppingAmplitudeSet.tabulate(&amplitudes, &indices, &numHoppingAmplitudes, &maxIndexSize);

	const int INDEX_RANK = 3;
	hsize_t indexDims[INDEX_RANK];
	indexDims[0] = numHoppingAmplitudes;
	indexDims[1] = 2; //Two indices per HoppingAmplitude
	indexDims[2] = maxIndexSize;
	const int AMPLITUDE_RANK = 1;
	hsize_t amplitudeDims[AMPLITUDE_RANK];
	amplitudeDims[0] = 2*numHoppingAmplitudes;	//2 because data is complex<double> interpreted as 2*double

	try{
		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeHoppingAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeHoppingAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeHoppingAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}

	delete [] amplitudes;
	delete [] indices;
}

void FileWriter::writeGeometry(
	const Geometry &geometry,
	string name,
	string path
){
	TBTKExit(
		"FileWriter::writeGeometry()",
		"This function is no longer supported.",
		"The FileReader and FileWriter are deprecated. Please use"
		<< " serialization instead."
	);
	//The coordinate data can no longer be easily stored by the FileWriter.
	//The code blow is therefore not valid any longer and commented out and
	//left here in case they are needed for future reference.

/*	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeGeometry()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int dimensions = geometry.getDimensions();
	int numSpecifiers = geometry.getNumSpecifiers();
	const double* coordinates = geometry.getCoordinates();
	const int* specifiers = geometry.getSpecifiers();
	int basisSize = geometry.getBasisSize();

	const int RANK = 2;
	hsize_t dDims[RANK];
	dDims[0] = basisSize;
	dDims[1] = dimensions;
	hsize_t sDims[RANK];
	sDims[0] = basisSize;
	sDims[1] = numSpecifiers;

	try{
		H5::Exception::dontPrint();
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
		if(numSpecifiers != 0){
			dataset.write(specifiers, PredType::NATIVE_INT);
		}
		else{
			int dummy[1];
			dataset.write(dummy, PredType::NATIVE_INT);
		}
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
	}*/
}

void FileWriter::writeIndexTree(
	const IndexTree &indexTree,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeIndexTree()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	std::vector<int> serializedIndices;
/*	IndexTree::Iterator it = indexTree.begin();
	while(!it.getHasReachedEnd()){
		Index index = it.getIndex();
		serializedIndices.push_back(index.getSize());
		for(unsigned int n = 0; n < index.getSize(); n++)
			serializedIndices.push_back(index.at(n));

		it.searchNext();
	}*/
	for(
		IndexTree::ConstIterator iterator = indexTree.cbegin();
		iterator != indexTree.cend();
		++iterator
	){
		Index index = *iterator;
		serializedIndices.push_back(index.getSize());
		for(unsigned int n = 0; n < index.getSize(); n++)
			serializedIndices.push_back(index.at(n));
	}

	const int RANK = 1;
	hsize_t dims[RANK] = {serializedIndices.size()};
	try{
		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		DataSpace dataspace = DataSpace(RANK, dims);
		DataSet dataset = DataSet(file.createDataSet(ss.str(), PredType::STD_I32BE, dataspace));
		dataset.write(serializedIndices.data(), PredType::NATIVE_INT);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeIndexTree()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeIndexTree()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeIndexTree()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::writeEigenValues(
	const Property::EigenValues &ev,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeEigenValues()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	const int RANK = 1;
	hsize_t dims[1];
	dims[0] = ev.getData().size();

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(RANK, dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(
			ev.getData().data(),
			PredType::NATIVE_DOUBLE
		);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::writeWaveFunctions(
	const Property::WaveFunctions &waveFunctions,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeWaveFunctions()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int attributes[2];
	attributes[0] = static_cast<int>(waveFunctions.getIndexDescriptor().getFormat());
	attributes[1] = static_cast<int>(waveFunctions.getStates().size());
	string attributeNames[2];
	attributeNames[0] = "Format";
	attributeNames[1] = "NumStates";
	stringstream ss;
	ss << name << "Attributes";
	writeAttributes(
		attributes,
		attributeNames,
		2,
		ss.str(),
		path
	);

	switch(waveFunctions.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		writeIndexTree(
			waveFunctions.getIndexDescriptor().getIndexTree(),
			ss.str(),
			path
		);

		const vector<unsigned int> &states = waveFunctions.getStates();
		const int STATES_RANK = 1;
		int statesDims[STATES_RANK] = {(int)states.size()};
		ss.str("");
		ss << name << "States";
		write((int*)states.data(), STATES_RANK, statesDims, ss.str(), path);

		const int RANK = 1;
		int dims[RANK] = {(int)waveFunctions.getData().size()};
		write(waveFunctions.getData().data(), RANK, dims, name, path);

		break;
	}
	default:
		TBTKExit(
			"FileWriter::writeWaveFunctions()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}
}

void FileWriter::writeDOS(const Property::DOS &dos, string name, string path){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeDOS()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	const int DOS_RANK = 1;
	hsize_t dos_dims[1];
	dos_dims[0] = dos.getResolution();

	double limits[2];
	limits[0] = dos.getUpperBound();
	limits[1] = dos.getLowerBound();
	const int LIMITS_RANK = 1;
	hsize_t limits_dims[1];
	limits_dims[0] = 2;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(DOS_RANK, dos_dims);
		DataSet dataset = DataSet(
			file.createDataSet(
				name,
				PredType::IEEE_F64BE,
				dataspace
			)
		);
		dataset.write(dos.getData().data(), PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataspace = DataSpace(LIMITS_RANK, limits_dims);
		Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
		attribute.write(PredType::NATIVE_DOUBLE, limits);
		dataspace.close();
		dataset.close();

		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::writeDensity(
	const Property::Density &density,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeDensity()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int attributes[1];
	attributes[0] = static_cast<int>(
		density.getIndexDescriptor().getFormat()
	);
	string attributeNames[1];
	attributeNames[0] = "Format";
	stringstream ss;
	ss << name << "Attributes";
	writeAttributes(
		attributes,
		attributeNames,
		1,
		ss.str(),
		path
	);

	switch(density.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Ranges:
	{
		int rank = density.getDimensions();
//		const int *dims = density.getRanges();
		vector<int> dims = density.getRanges();

		hsize_t density_dims[rank];
		for(int n = 0; n < rank; n++)
			density_dims[n] = dims[n];

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
			H5File file(filename, H5F_ACC_RDWR);

			DataSpace dataspace = DataSpace(rank, density_dims);
			DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
			dataset.write(density.getData().data(), PredType::NATIVE_DOUBLE);
			dataspace.close();
			dataset.close();
			file.close();
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeDensity()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeDensity()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeDensity()",
				"While writing to " << name << ".",
				""
			);
		}
		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		writeIndexTree(
			density.getIndexDescriptor().getIndexTree(),
			ss.str(),
			path
		);

		const int RANK = 1;
		int dims[RANK] = {(int)density.getData().size()};
		write(density.getData().data(), RANK, dims, name, path);

		break;
	}
	default:
		TBTKExit(
			"FileWriter::writeDensity()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}
}

void FileWriter::writeMagnetization(
	const Property::Magnetization &magnetization,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeMagnetization()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int attributes[1];
	attributes[0] = static_cast<int>(magnetization.getIndexDescriptor().getFormat());
	string attributeNames[1];
	attributeNames[0] = "Format";
	stringstream ss;
	ss << name << "Attributes";
	writeAttributes(
		attributes,
		attributeNames,
		1,
		ss.str(),
		path
	);

	switch(magnetization.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Ranges:
	{
		int rank = magnetization.getDimensions();
//		const int *dims = magnetization.getRanges();
		vector<int> dims = magnetization.getRanges();
		const std::vector<SpinMatrix> &data = magnetization.getData();

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
		for(int n = 0; n < mag_size/4; n++){
			mag_decomposed[8*n+0] = real(data[n].at(0, 0));
			mag_decomposed[8*n+1] = imag(data[n].at(0, 0));
			mag_decomposed[8*n+2] = real(data[n].at(0, 1));
			mag_decomposed[8*n+3] = imag(data[n].at(0, 1));
			mag_decomposed[8*n+4] = real(data[n].at(1, 0));
			mag_decomposed[8*n+5] = imag(data[n].at(1, 0));
			mag_decomposed[8*n+6] = real(data[n].at(1, 1));
			mag_decomposed[8*n+7] = imag(data[n].at(1, 1));
		}
		mag_dims[rank+1] = 2;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
			H5File file(filename, H5F_ACC_RDWR);

			DataSpace dataspace = DataSpace(rank+2, mag_dims);
			DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
			dataset.write(mag_decomposed, PredType::NATIVE_DOUBLE);
			dataspace.close();
			dataset.close();
			file.close();
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeMagnetization()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeMagnetization()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeMagnetization()",
				"While writing to " << name << ".",
				""
			);
		}

		delete [] mag_decomposed;

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		writeIndexTree(
			magnetization.getIndexDescriptor().getIndexTree(),
			ss.str(),
			path
		);

		const int RANK = 1;
		int dims[RANK] = {4*(int)magnetization.getData().size()};
		const std::vector<SpinMatrix> &data = magnetization.getData();
		complex<double> *data_internal = new complex<double>[dims[0]];
		for(int n = 0; n < dims[0]/4; n++){
			data_internal[4*n + 0] = data[n].at(0, 0);
			data_internal[4*n + 1] = data[n].at(0, 1);
			data_internal[4*n + 2] = data[n].at(1, 0);
			data_internal[4*n + 3] = data[n].at(1, 1);
		}

		write(data_internal, RANK, dims, name, path);

		break;
	}
	default:
		TBTKExit(
			"FileWriter::writeMagnetization()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}
}

void FileWriter::writeLDOS(
	const Property::LDOS &ldos,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeLDOS()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int intAttributes[2];
	intAttributes[0] = static_cast<int>(ldos.getIndexDescriptor().getFormat());
	intAttributes[1] = ldos.getResolution();
	string intAttributeNames[2];
	intAttributeNames[0] = "Format";
	intAttributeNames[1] = "Resolution";
	stringstream ss;
	ss << name << "IntAttributes";
	writeAttributes(
		intAttributes,
		intAttributeNames,
		2,
		ss.str(),
		path
	);

	double doubleAttributes[2];
	doubleAttributes[0] = ldos.getLowerBound();
	doubleAttributes[1] = ldos.getUpperBound();
	string doubleAttributeNames[2];
	doubleAttributeNames[0] = "LowerBound";
	doubleAttributeNames[1] = "UpperBound";
	ss.str("");
	ss << name << "DoubleAttributes";
	writeAttributes(
		doubleAttributes,
		doubleAttributeNames,
		2,
		ss.str(),
		path
	);

	switch(ldos.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Ranges:
	{
		int rank = ldos.getDimensions();
//		const int *dims = ldos.getRanges();
		vector<int> dims = ldos.getRanges();

		hsize_t ldos_dims[rank+1];//Last dimension is for energy
		for(int n = 0; n < rank; n++)
			ldos_dims[n] = dims[n];
		ldos_dims[rank] = ldos.getResolution();

		double limits[2];
		limits[0] = ldos.getUpperBound();
		limits[1] = ldos.getLowerBound();
		const int LIMITS_RANK = 1;
		hsize_t limits_dims[1];
		limits_dims[0] = 2;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
			H5File file(filename, H5F_ACC_RDWR);

			DataSpace dataspace = DataSpace(rank+1, ldos_dims);
			DataSet dataset = DataSet(
				file.createDataSet(
					name,
					PredType::IEEE_F64BE,
					dataspace
				)
			);
			dataset.write(
				ldos.getData().data(),
				PredType::NATIVE_DOUBLE
			);
			dataspace.close();

			dataspace = DataSpace(LIMITS_RANK, limits_dims);
			Attribute attribute = dataset.createAttribute("UpLowLimits", PredType::IEEE_F64BE, dataspace);
			attribute.write(PredType::NATIVE_DOUBLE, limits);
			dataspace.close();
			dataset.close();

			file.close();
		}
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeLDOS()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeLDOS()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeLDOS()",
				"While writing to " << name << ".",
				""
			);
		}

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		writeIndexTree(
			ldos.getIndexDescriptor().getIndexTree(),
			ss.str(),
			path
		);

		const int RANK = 1;
		int dims[RANK] = {(int)ldos.getData().size()};
		write(ldos.getData().data(), RANK, dims, name, path);

		break;
	}
	default:
		TBTKExit(
			"FileWriter::writeLDOS()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}
}

void FileWriter::writeSpinPolarizedLDOS(
	const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeSpinPolarizedLDOS()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	int intAttributes[2];
	intAttributes[0] = static_cast<int>(spinPolarizedLDOS.getIndexDescriptor().getFormat());
	intAttributes[1] = spinPolarizedLDOS.getResolution();
	string intAttributeNames[2];
	intAttributeNames[0] = "Format";
	intAttributeNames[1] = "Resolution";
	stringstream ss;
	ss << name << "IntAttributes";
	writeAttributes(
		intAttributes,
		intAttributeNames,
		2,
		ss.str(),
		path
	);

	double doubleAttributes[2];
	doubleAttributes[0] = spinPolarizedLDOS.getLowerBound();
	doubleAttributes[1] = spinPolarizedLDOS.getUpperBound();
	string doubleAttributeNames[2];
	doubleAttributeNames[0] = "LowerBound";
	doubleAttributeNames[1] = "UpperBound";
	ss.str("");
	ss << name << "DoubleAttributes";
	writeAttributes(
		doubleAttributes,
		doubleAttributeNames,
		2,
		ss.str(),
		path
	);

	switch(spinPolarizedLDOS.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Ranges:
	{
		int rank = spinPolarizedLDOS.getDimensions();
//		const int *dims = spinPolarizedLDOS.getRanges();
		vector<int> dims = spinPolarizedLDOS.getRanges();
		const std::vector<SpinMatrix> &data
			= spinPolarizedLDOS.getData();

		const int NUM_MATRIX_ELEMENTS = 4;
		hsize_t sp_ldos_dims[rank+2];//Three last dimensions are for energy, spin components, and real/imaginary decomposition.
		for(int n = 0; n < rank; n++)
			sp_ldos_dims[n] = dims[n];
		sp_ldos_dims[rank] = spinPolarizedLDOS.getResolution();
		sp_ldos_dims[rank+1] = NUM_MATRIX_ELEMENTS;

		int sp_ldos_size = 1;
		for(int n = 0; n < rank+2; n++)
			sp_ldos_size *= sp_ldos_dims[n];
		double *sp_ldos_decomposed;
		sp_ldos_decomposed = new double[2*sp_ldos_size];
		for(int n = 0; n < sp_ldos_size/4; n++){
			sp_ldos_decomposed[8*n+0] = real(data[n].at(0, 0));
			sp_ldos_decomposed[8*n+1] = imag(data[n].at(0, 0));
			sp_ldos_decomposed[8*n+2] = real(data[n].at(0, 1));
			sp_ldos_decomposed[8*n+3] = imag(data[n].at(0, 1));
			sp_ldos_decomposed[8*n+4] = real(data[n].at(1, 0));
			sp_ldos_decomposed[8*n+5] = imag(data[n].at(1, 0));
			sp_ldos_decomposed[8*n+6] = real(data[n].at(1, 1));
			sp_ldos_decomposed[8*n+7] = imag(data[n].at(1, 1));
		}

		sp_ldos_dims[rank+2] = 2;

		double limits[2];
		limits[0] = spinPolarizedLDOS.getUpperBound();
		limits[1] = spinPolarizedLDOS.getLowerBound();
		const int LIMITS_RANK = 1;
		hsize_t limits_dims[1];
		limits_dims[0] = 2;

		try{
			stringstream ss;
			ss << path;
			if(path.back() != '/')
				ss << "/";
			ss << name;

			H5::Exception::dontPrint();
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
		catch(FileIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeSpinPolarizedLDOS()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSetIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeSpinPolarizedLDOS()",
				"While writing to " << name << ".",
				""
			);
		}
		catch(DataSpaceIException &error){
			Streams::log << error.getCDetailMsg() << "\n";
			TBTKExit(
				"FileWriter::writeSpinPolarizedLDOS()",
				"While writing to " << name << ".",
				""
			);
		}

		delete [] sp_ldos_decomposed;

		break;
	}
	case IndexDescriptor::Format::Custom:
	{
		stringstream ss;
		ss << name << "IndexTree";
		writeIndexTree(
			spinPolarizedLDOS.getIndexDescriptor().getIndexTree(),
			ss.str(),
			path
		);

		const int RANK = 1;
		int dims[RANK] = {4*(int)spinPolarizedLDOS.getData().size()};
		const std::vector<SpinMatrix> &data
			= spinPolarizedLDOS.getData();
		complex<double> *data_internal = new complex<double>[dims[0]];
		for(int n = 0; n < dims[0]/4; n++){
			data_internal[4*n + 0] = data[n].at(0, 0);
			data_internal[4*n + 1] = data[n].at(0, 1);
			data_internal[4*n + 2] = data[n].at(1, 0);
			data_internal[4*n + 3] = data[n].at(1, 1);
		}

		write(data_internal, RANK, dims, name, path);

		break;
	}
	default:
		TBTKExit(
			"FileWriter::writeLDOS()",
			"Storage format not supported.",
			"This should never happen, contact the developer."
		);
	}
}

void FileWriter::write(
	const int *data,
	int rank,
	const int *dims,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::write()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

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

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, data_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::STD_I32BE, dataspace));
		dataset.write(data, PredType::NATIVE_INT);
		dataspace.close();

		dataset.close();
		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::write(
	const double *data,
	int rank,
	const int *dims,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::write()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

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

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(rank, data_dims);
		DataSet dataset = DataSet(file.createDataSet(name, PredType::IEEE_F64BE, dataspace));
		dataset.write(data, PredType::NATIVE_DOUBLE);
		dataspace.close();

		dataset.close();
		file.close();
	}
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::write(
	const complex<double> *data,
	int rank,
	const int *dims,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::write()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	unsigned int size = 1;
	for(unsigned int n = 0; n < (unsigned int)rank; n++)
		size *= dims[n];

	double *realData = new double[size];
	double *imagData = new double[size];
	for(unsigned int n = 0; n < size; n++){
		realData[n] = real(data[n]);
		imagData[n] = imag(data[n]);
	}

	stringstream ss;
	ss << name << "Real";
	write(realData, rank, dims, ss.str(), path);
	ss.str("");
	ss << name << "Imag";
	write(imagData, rank, dims, ss.str(), path);

	delete [] realData;
	delete [] imagData;
}

void FileWriter::writeAttributes(
	const int *attributes,
	const string *attribute_names,
	int num,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeAttributes()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

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

		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << "",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::writeAttributes(
	const double *attributes,
	const string *attribute_names,
	int num,
	string name,
	string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeAttributes()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

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

		H5::Exception::dontPrint();
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
	catch(FileIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
}

bool FileWriter::exists(){
	ifstream fin(filename);
	bool exists = fin.good();
	fin.close();

	return exists;
}

void FileWriter::writeParameterSet(
	const ParameterSet *parameterSet,
	std::string name,
	std::string path
){
	TBTKAssert(
		path.compare("/") == 0,
		"FileWriter::writeParameterSet()",
		"'path' not yet supported.",
		"Only use the default path value \"/\"."
	);

	init();

	const int ATTRIBUTES_RANK = 0;
	const hsize_t *attribute_dims = NULL;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		H5::Exception::dontPrint();
		H5File file(filename, H5F_ACC_RDWR);

		DataSpace dataspace = DataSpace(ATTRIBUTES_RANK, attribute_dims);
		DataSet dataset = DataSet(file.createDataSet(name + "Int", PredType::STD_I64BE, dataspace));

		for(int n = 0; n < parameterSet->getNumInt(); n++){
			Attribute attribute = dataset.createAttribute(parameterSet->getIntName(n), PredType::STD_I64BE, dataspace);
			int value = parameterSet->getIntValue(n);
			attribute.write(PredType::NATIVE_INT, &value);
		}

		dataset = DataSet(file.createDataSet(name + "Double", PredType::IEEE_F64BE, dataspace));

		for(int n = 0; n < parameterSet->getNumDouble(); n++){
			Attribute attribute = dataset.createAttribute(parameterSet->getDoubleName(n), PredType::IEEE_F64BE, dataspace);
			double value = parameterSet->getDoubleValue(n);
			attribute.write(PredType::NATIVE_DOUBLE, &value);
		}

		const int COMPLEX_RANK = 1;
		const hsize_t complex_dims[COMPLEX_RANK] = {2};
		ArrayType complexDataType(PredType::NATIVE_DOUBLE, COMPLEX_RANK, complex_dims);
		dataset = DataSet(file.createDataSet(name + "Complex", PredType::IEEE_F64BE, dataspace));

		for(int n = 0; n < parameterSet->getNumComplex(); n++){
			Attribute attribute = dataset.createAttribute(parameterSet->getComplexName(n), complexDataType, dataspace);
			complex<double> complexValue = parameterSet->getComplexValue(n);
			double value[2] = {real(complexValue), imag(complexValue)};
			attribute.write(complexDataType, value);
		}

		dataset = DataSet(file.createDataSet(name + "String", PredType::PredType::C_S1, dataspace));

		for(int n = 0; n < parameterSet->getNumString(); n++){
			string value = parameterSet->getStringValue(n);
			StrType strDataType(PredType::C_S1, value.length());
			const H5std_string strWriteBuf(value);
			Attribute attribute = dataset.createAttribute(parameterSet->getStringName(n), strDataType, dataspace);
			attribute.write(strDataType, strWriteBuf);
		}

		dataset = DataSet(file.createDataSet(name + "Bool", PredType::STD_I64BE, dataspace));

		for(int n = 0; n < parameterSet->getNumBool(); n++){
			Attribute attribute = dataset.createAttribute(parameterSet->getBoolName(n), PredType::STD_I64BE, dataspace);
			int value = parameterSet->getBoolValue(n);
			attribute.write(PredType::NATIVE_INT, &value);
		}

		dataspace.close();
		dataset.close();
		file.close();
	}
	catch(FileIException &error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException &error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException &error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
}

};	//End of namespace TBTK
