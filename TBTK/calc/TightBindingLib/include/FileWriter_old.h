#ifndef COM_DAFER45_TIGHT_BINDING_LIB_FILE_WRITER
#define COM_DAFER45_TIGHT_BINDING_LIB_FILE_WRITER

#include "FileIO.h"
#include <fstream>

class FileWriter{
public:
	static void writeEV(	double *ev,
				int size,
				std::string filename = "EV.bin",
				std::string path = "./"
	);
	static void writeDOS(	double *ev,
				double u_lim,
				double l_lim,
				int resolution,
				std::string filename = "DOS.bin",
				std::string path = "./"
	);
//private:
	static void writeHeader(std::ofstream &fout, TBFileType tbFileType);

	static void writeShort(std::ofstream &fout, short s);
	static void writeInt(std::ofstream &fout, int i);
	static void writeFloat(std::ofstream &fout, float f);
	static void writeDouble(std::ofstream &fout, double d);
};

#endif

