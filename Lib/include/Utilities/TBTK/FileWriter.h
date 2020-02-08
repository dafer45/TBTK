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
 *  @file FileWriter.h
 *  @brief Writes data to file
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#ifndef COM_DAFER45_TBTK_FILE_WRITER
#define COM_DAFER45_TBTK_FILE_WRITER

#include "TBTK/Model.h"
#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/Geometry.h"
#include "TBTK/ParameterSet.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/SpectralFunction.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Property/WaveFunctions.h"
#include <fstream>
#include <stdio.h>

namespace TBTK{

/** Writes data to a .hdf5-file. The default file name is TBTKResults.h5. Can
 *  be used to write custom n-dimensional arrays of data and parameters to
 *  datasets with custom names. It can also be used to write data such as
 *  eigenvalues, DOS, Density etc. extracted by the PropertyExtractor. In the
 *  later case the data can immediately be plotted using the bundled python
 *  plotting scripts.
 */
class FileWriter{
public:
	/** Write model to file. */
	static void writeModel(
		const Model &model,
		std::string name = "Model",
		std::string path = "/"
	);

	/** Experimental. Write HoppingAmplitudeSet to file. */
	static void writeHoppingAmplitudeSet(
		const HoppingAmplitudeSet &hoppingAmplitudeSet,
		std::string name = "HoppingAmplitudeSet",
		std::string path = "/"
	);

	static void writeGeometry(
		const Geometry &geometry,
		std::string name = "Geometry",
		std::string path = "/"
	);

	static void writeIndexTree(
		const IndexTree &indexTree,
		std::string name = "IndexTree",
		std::string path = "/"
	);

	/** Write eigenvalues to file. */
	static void writeEigenValues(
		const Property::EigenValues &ev,
		std::string name = "EigenValues",
		std::string path = "/"
	);

	/** Write wave function to file. */
	static void writeWaveFunctions(
		const Property::WaveFunctions &waveFunctions,
		std::string name = "WaveFunctions",
		std::string path = "/"
	);

	/** Write density of states to file. */
	static void writeDOS(
		const Property::DOS &dos,
		std::string name = "DOS",
		std::string path = "/"
	);

	/** Write density to file. */
	static void writeDensity(
		const Property::Density &density,
		std::string name = "Density",
		std::string path = "/"
	);

	/** Write magnetization to file. */
	static void writeMagnetization(
		const Property::Magnetization &magnetization,
		std::string name = "Magnetization",
		std::string path = "/"
	);

	/** Write local density of states to file. */
	static void writeLDOS(
		const Property::LDOS &ldos,
		std::string name = "LDOS",
		std::string path = "/"
	);

	/** Write spin-polarized local density of states to file. */
	static void writeSpinPolarizedLDOS(
		const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
		std::string name = "SpinPolarizedLDOS",
		std::string path = "/"
	);

	/** Write custom n-dimensional arrays to file of type int. */
	static void write(
		const int *data,
		int rank,
		const int *dims,
		std::string name,
		std::string path = "/"
	);

	/** Write custom n-dimensional arrays to file of type double. */
	static void write(
		const double *data,
		int rank,
		const int *dims,
		std::string name,
		std::string path = "/"
	);

	/** Write custom n-dimensional arrays to file of type complex<double>.
	 */
	static void write(
		const std::complex<double> *data,
		int rank,
		const int *dims,
		std::string name,
		std::string path = "/"
	);

	/**Write custom attributes to file of type int. */
	static void writeAttributes(
		const int *attributes,
		const std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Write custom attributes to file of type double. */
	static void writeAttributes(
		const double *attributes,
		const std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Write ParamterSet to file.*/
	static void writeParameterSet(
		const ParameterSet *parameterSet,
		std::string name = "ParameterSet",
		std::string path = "/"
	);

	/** Set output file name. Default is TBTKResults.h5. */
	static void setFileName(std::string filename);

	/** Remove any file from the current folder with the file name set by
	 *  FileWriter::setFileName*/
	static void clear();

	/** Returns true if current input file exists. */
	static bool exists();
private:
	/** Is set to true if the file has been opened. */
	static bool isInitialized;

	/** Open file and prepare for writing. */
	static void init();

	/** File name of file to write to. */
	static std::string filename;
};

inline void FileWriter::setFileName(std::string filename){
	FileWriter::filename = filename;
	isInitialized = false;
}

inline void FileWriter::clear(){
	remove(filename.c_str());
	isInitialized = false;
}

};	//End of namespace TBTK

#endif

