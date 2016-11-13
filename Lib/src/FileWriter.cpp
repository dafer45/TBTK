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

#include "FileWriter.h"

#include <string>
#include <sstream>
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

	writeAmplitudeSet(model->getAmplitudeSet(), ss.str(), path);

	ss.str("");
	ss << name << "Geometry";
	if(model->getGeometry() != NULL)
		writeGeometry(model->getGeometry(), ss.str(), path);


	const int NUM_DOUBLE_ATTRIBUTES = 2;
	ss.str("");
	ss << name << "DoubleAttributes";
	double doubleAttributes[NUM_DOUBLE_ATTRIBUTES] = {model->getTemperature(), model->getChemicalPotential()};
	string doubleAttributeNames[NUM_DOUBLE_ATTRIBUTES] = {"Temperature", "ChemicalPotential"};
	writeAttributes(doubleAttributes, doubleAttributeNames, NUM_DOUBLE_ATTRIBUTES, ss.str());

	const int NUM_INT_ATTRIBUTES = 1;
	ss.str("");
	ss << name << "IntAttributes";
	int intAttributes[NUM_INT_ATTRIBUTES] = {static_cast<int>(model->getStatistics())};
	string intAttributeNames[NUM_INT_ATTRIBUTES] = {"Statistics"};
	writeAttributes(intAttributes, intAttributeNames, NUM_INT_ATTRIBUTES, ss.str());
}

void FileWriter::writeAmplitudeSet(
	AmplitudeSet *amplitudeSet,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAmplitudeSet()",
			"While writing to " << name << ".",
			""
		);
	}

	delete [] amplitudes;
	delete [] indices;
}

void FileWriter::writeGeometry(
	const Geometry *geometry,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeGeometry()",
			"While writing to " << name << ".",
			""
		);
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

void FileWriter::writeEigenValues(
	const Property::EigenValues *ev,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeEigenValues()",
			"While writing to " << name << ".",
			""
		);
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDOS()",
			"While writing to " << name << ".",
			""
		);
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

void FileWriter::writeDensity(
	const Property::Density *density,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDensity()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDensity()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeDensity()",
			"While writing to " << name << ".",
			""
		);
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

void FileWriter::writeMagnetization(
	const Property::Magnetization *magnetization,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeMagnetization()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeMagnetization()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeMagnetization()",
			"While writing to " << name << ".",
			""
		);
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

void FileWriter::writeLDOS(
	const Property::LDOS *ldos,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeLDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeLDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeLDOS()",
			"While writing to " << name << ".",
			""
		);
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

void FileWriter::writeSpinPolarizedLDOS(
	const Property::SpinPolarizedLDOS *spinPolarizedLDOS,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeSpinPolarizedLDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeSpinPolarizedLDOS()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeSpinPolarizedLDOS()",
			"While writing to " << name << ".",
			""
		);
	}

	delete [] sp_ldos_decomposed;
}

void FileWriter::write(
	const double *data,
	int rank,
	const int *dims,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::write()",
			"While writing to " << name << ".",
			""
		);
	}
}

void FileWriter::writeAttributes(
	const int *attributes,
	const string *attribute_names,
	int num,
	string name,
	string path
){
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << "",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
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
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
		TBTKExit(
			"FileWriter::writeAttributes()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		Util::Streams::log << error.getCDetailMsg() << "\n";
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
	const Util::ParameterSet *parameterSet,
	std::string name,
	std::string path
){
	init();

	const int ATTRIBUTES_RANK = 0;
	const hsize_t *attribute_dims = NULL;

	try{
		stringstream ss;
		ss << path;
		if(path.back() != '/')
			ss << "/";
		ss << name;

		Exception::dontPrint();
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
	catch(FileIException error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSetIException error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
	catch(DataSpaceIException error){
		TBTKExit(
			"FileWriter::writeParameterSet()",
			"While writing to " << name << ".",
			""
		);
	}
}

};	//End of namespace TBTK
