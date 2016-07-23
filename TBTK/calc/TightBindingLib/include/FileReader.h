/** @package TBTKcalc
 *  @file FileReader.h
 *  @brief Reads data from file
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FILE_READER
#define COM_DAFER45_TBTK_FILE_READER

#include "AmplitudeSet.h"
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
	/** Experimental. Read AmplitudeSet from file. */
	static void readAmplitudeSet(
		AmplitudeSet **amplitudeSet,
		std::string name = "AmplitudeSet",
		std::string path = "/"
	);

	/** Read eigenvalues from file. */
	static void readEigenValues(
		double **ev,
		int *size,
		std::string name = "EV",
		std::string path = "/"
	);

	/** Read density of states from file. */
	static void readDOS(
		double **dos,
		double *l_lim,
		double *u_lim,
		int *resolution,
		std::string name = "DOS",
		std::string path = "/"
	);

	/** Read density from file. */
	static void readDensity(
		double **density,
		int *rank,
		int **dims,
		std::string name = "Density",
		std::string path = "/"
	);

	/** Read magnetization from file. */
	static void readMAG(
		std::complex<double> **mag,
		int *rank,
		int **dims,
		std::string name = "MAG",
		std::string path = "/"
	);

	/** Read local density of states from file. */
	static void readLDOS(
		double **ldos,
		int *rank,
		int **dims,
		double *l_lim,
		double *u_lim,
		int *resolution,
		std::string name = "LDOS",
		std::string path = "/"
	);

	/** Read spin-polarized local density of states from file. */
	static void readSP_LDOS(
		std::complex<double> **sp_ldos,
		int *rank,
		int **dims,
		double *l_lim,
		double *u_lim,
		int *resolution,
		std::string name = "SP_LDOS",
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
private:
	/** Is set to true if the file has been opened. */
	static bool isInitialized;

	/** File name of file to read from. */
	static std::string filename;
};

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

