/** @package TBTKcalc
 *  @file FileWriter.h
 *  @brief Writes data to file
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FILE_WRITER
#define COM_DAFER45_TBTK_FILE_WRITER

#include "AmplitudeSet.h"
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
	/** Experimental. Write AmplitudeSet to file. */
	static void writeAmplitudeSet(
		AmplitudeSet *amplitudeSet,
		std::string name = "AmplitudeSet",
		std::string path = "/"
	);

	/** Write eigenvalues to file. */
	static void writeEigenValues(
		double *ev,
		int size,
		std::string name = "EV",
		std::string path = "/"
	);

	/** Write density of states to file. */
	static void writeDOS(
		double *dos,
		double l_lim,
		double u_lim,
		int resolution,
		std::string name = "DOS",
		std::string path = "/"
	);

	/** Write density to file. */
	static void writeDensity(
		double *density,
		int rank,
		int *dims,
		std::string name = "Density",
		std::string path = "/"
	);

	/** Write magnetization to file. */
	static void writeMAG(
		std::complex<double> *mag,
		int rank,
		int *dims,
		std::string name = "MAG",
		std::string path = "/"
	);
/*	static void writeMAG(
		double *mag,
		int rank,
		int *dims,
		std::string name = "MAG",
		std::string path = "/"
	);*/

	/** Write local density of states to file. */
	static void writeLDOS(
		double *ldos,
		int rank,
		int *dims,
		double l_lim,
		double u_lim,
		int resolution,
		std::string name = "LDOS",
		std::string path = "/"
	);

	/** Write spin-polarized local density of states to file. */
	static void writeSP_LDOS(
		std::complex<double> *sp_ldos,
		int rank,
		int *dims,
		double l_lim,
		double u_lim,
		int resolution,
		std::string name = "SP_LDOS",
		std::string path = "/"
	);
/*	static void writeSP_LDOS(
		double *sp_ldos,
		int rank,
		int *dims,
		double u_lim,
		double l_lim,
		int resolution,
		std::string name = "SP_LDOS",
		std::string path = "/"
	);*/

	/** Write custom n-dimensional arrays to file of type double. */
	static void write(
		double *data,
		int rank,
		int *dims,
		std::string name,
		std::string path = "/"
	);

	/**Write custom attributes to file of type int. */
	static void writeAttributes(
		int *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Write custom attributes to file of type double. */
	static void writeAttributes(
		double *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);

	/** Set output file name. Default is TBTKResults.h5. */
	static void setFileName(std::string filename);

	/** Remove any file from the current folder with the file name set by
	 *  FileWriter::setFileName*/
	static void clear();
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

