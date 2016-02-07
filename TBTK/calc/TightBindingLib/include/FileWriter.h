#ifndef COM_DAFER45_TBTK_FILE_WRITER
#define COM_DAFER45_TBTK_FILE_WRITER

#include "FileIO.h"
#include "AmplitudeSet.h"
#include <fstream>
#include <stdio.h>

class FileWriter{
public:
	static void writeAmplitudeSet(
		AmplitudeSet *amplitudeSet,
		std::string name = "AmplitudeSet",
		std::string path = "/"
	);
	static void writeEV(
		double *ev,
		int size,
		std::string name = "EV",
		std::string path = "/"
	);
	static void writeDOS(
		double *ev,
		double u_lim,
		double l_lim,
		int resolution,
		std::string name = "DOS",
		std::string path = "/"
	);
	static void writeDensity(
		double *density,
		int rank,
		int *dims,
		std::string name = "Density",
		std::string path = "/"
	);
	static void writeMAG(
		double *mag,
		int rank,
		int *dims,
		std::string name = "MAG",
		std::string path = "/"
	);
	static void writeSP_LDOS(
		double *sp_ldos,
		int rank,
		int *dims,
		double u_lim,
		double l_lim,
		int resolution,
		std::string name = "SP_LDOS",
		std::string path = "/"
	);
	static void write(
		double *data,
		int rank,
		int *dims,
		std::string name,
		std::string path = "/"
	);
	static void writeAttributes(
		int *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);
	static void writeAttributes(
		double *attributes,
		std::string *attribute_names,
		int num,
		std::string name,
		std::string path = "/"
	);
	static void setFileName(std::string filename);
	static void clear();
private:
	static bool isInitialized;
	static void init();
	static std::string filename;
/*	static void writeHeader(std::ofstream &fout, TBFileType tbFileType);

	static void writeShort(std::ofstream &fout, short s);
	static void writeInt(std::ofstream &fout, int i);
	static void writeFloat(std::ofstream &fout, float f);
	static void writeDouble(std::ofstream &fout, double d);*/
};

inline void FileWriter::setFileName(std::string filename){
	FileWriter::filename = filename;
}

inline void FileWriter::clear(){
	remove(filename.c_str());
	isInitialized = false;
}

#endif

