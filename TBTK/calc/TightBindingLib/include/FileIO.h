#ifndef COM_DAFER45_TIGHT_BINDING_VISUALIZER_FILE_IO
#define COM_DAFER45_TIGHT_BINDING_VISUALIZER_FILE_IO

#include <fstream>
#include <complex>

enum TBFileType {TB_FILE_TYPE_EV, TB_FILE_TYPE_DOS, TB_FILE_TYPE_LDOS};

class TBFileHeader{
public:
	char magic[4];
	int file_version;
	int software_version;
	int checksum;
	TBFileType fileType;
private:
};

class TBFile{
public:
	TBFileHeader header;
	union {
		int *idata;
		float *fdata;
		double *ddata;
		std::complex<double> *cdata;
	};
private:
};

#endif
