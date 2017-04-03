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

/** @package TBTKcalc
 *  @file FileReader.h
 *  @brief Reads data from file
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#ifndef COM_DAFER45_TBTK_FILE_READER
#define COM_DAFER45_TBTK_FILE_READER

#include "HoppingAmplitudeSet.h"
#include "Geometry.h"
#include "EigenValues.h"
#include "DOS.h"
#include "Density.h"
#include "Magnetization.h"
#include "Model.h"
#include "LDOS.h"
#include "SpectralFunction.h"
#include "SpinPolarizedLDOS.h"
#include "ParameterSet.h"
#include "WaveFunction.h"

#include <fstream>
#include <stdio.h>

namespace TBTK{

/** Reads data from a .hdf5-file. The default file name is TBTKResults.h5. Can
 *  be used to read custom n-dimensional arrays of data and parameters from
 *  datasets with custom names. It can also be used to read data such as
 *  eigenvalues, DOS, Density etc. written by the FileWriter.
 */
class FileReader{
public:
	/** Read model from file. */
	static Model* readModel(
		std::string name = "Model",
		std::string path = "/"
	);

	/** Experimental. Read HoppingAmplitudeSet from file. */
	static HoppingAmplitudeSet* readHoppingAmplitudeSet(
		std::string name = "HoppingAmplitudeSet",
		std::string path = "/"
	);

	/** Read geometry from file. */
	static Geometry* readGeometry(
		Model *model,
		std::string name = "Geometry",
		std::string path = "/"
	);

	/** Read IndexTree from file. */
	static IndexTree* readIndexTree(
		std::string name = "IndexTree",
		std::string path = "/"
	);

	/** Read eigenvalues from file. */
	static Property::EigenValues* readEigenValues(
		std::string name = "EigenValues",
		std::string path = "/"
	);

	/** Read WaveFunction from file. */
	static Property::WaveFunction* readWaveFunction(
		std::string name = "WaveFunction",
		std::string pat = "/"
	);

	/** Read density of states from file. */
	static Property::DOS* readDOS(
		std::string name = "DOS",
		std::string path = "/"
	);

	/** Read density from file. */
	static Property::Density* readDensity(
		std::string name = "Density",
		std::string path = "/"
	);

	/** Read magnetization from file. */
	static Property::Magnetization* readMagnetization(
		std::string name = "Magnetization",
		std::string path = "/"
	);

	/** Read local density of states from file. */
	static Property::LDOS* readLDOS(
		std::string name = "LDOS",
		std::string path = "/"
	);

	/** Read spectral function from file. */
	static Property::SpectralFunction* readSpectralFunction(
		std::string name = "SpectralFunction",
		std::string path = "/"
	);

	/** Read spin-polarized local density of states from file. */
	static Property::SpinPolarizedLDOS* readSpinPolarizedLDOS(
		std::string name = "SpinPolarizedLDOS",
		std::string path = "/"
	);

	/** Read ParameterSet from file. */
	static ParameterSet* readParameterSet(
		std::string name = "ParameterSet",
		std::string path = "/"
	);

	/** Read custom n-dimensional arrays from file of type int. */
	static void read(
		int **data,
		int *rank,
		int **dims,
		std::string name,
		std::string path = "/"
	);

	/** Read custom n-dimensional arrays from file of type double. */
	static void read(
		double **data,
		int *rank,
		int **dims,
		std::string name,
		std::string path = "/"
	);

	/** Read custom n-dimensional arrays from file of type complex<double>.
	 */
	static void read(
		std::complex<double> **data,
		int *rank,
		int **dims,
		std::string name,
		std::string path = "/"
	);

	/** Read custom attributes from file of type int. */
	static void readAttributes(
		int *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Read custom attributes from file of type double. */
	static void readAttributes(
		double *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Set input file name. Default is TBTKResults.h5. */
	static void setFileName(std::string filename);

	/** Remove any file from the current folder with the file name set by
	 *  FileReader::setFileName*/
	static void clear();

	/** Returns true if current input file exists. */
	static bool exists();
private:
	/** Is set to true if the file has been opened. */
	static bool isInitialized;

	/** File name of file to read from. */
	static std::string filename;
};

inline Property::SpectralFunction* FileReader::readSpectralFunction(
	std::string name,
	std::string path
){
	return reinterpret_cast<Property::SpectralFunction*>(readLDOS(name, path));
}

inline void FileReader::setFileName(std::string filename){
	FileReader::filename = filename;
	isInitialized = false;
}

inline void FileReader::clear(){
	remove(filename.c_str());
	isInitialized = false;
}

};	//End of namespace TBTK

#endif

