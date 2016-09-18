
#ifndef COM_DAFER45_TBTK_FILE_PARSER
#define COM_DAFER45_TBTK_FILE_PARSER

#include "Model.h"

#include <string>
#include <fstream>
#include <sstream>

namespace TBTK{

class FileParser{
public:
	enum class AmplitudeMode{ALL, ALL_EXCEPT_HC};

	static void writeModel(
		Model *model,
		std::string fileName,
		AmplitudeMode amplitudeMode,
		std::string description
	);

	static Model* readModel(std::string fileName);
private:
	static void openOutput(std::string fileName);

	static void closeOutput();

	static void readInput(std::string fileName);

	//Write
	static void writeLineBreaks(int numLineBreaks);

	static void writeTabs(int numTabs);

	static void write(std::complex<double> value);

	static void write(const Index &index);

	static void writeCoordinates(const double *coordinates, int numCoordinates);

	static void writeSpecifiers(const int *specifiers, int numSpecifiers);

	static void writeDescription(std::string description);

	static void writeAmplitudes(Model *model, AmplitudeMode amplitudeMode);

	static void writeGeometry(Model *model);

	//Read
	static void removeComments();

//	static void eatWhiteSpace();

	static int readParameter(std::string parameterName, std::string parentStructure);

	static void readAmplitudes(Model *model);

	static void readGeometry(Model *model);

//	static int readAmplitudeMode();

	static HoppingAmplitude* readHoppingAmplitude();

	static Index* readIndex();

	static void readCoordinates(std::vector<double> *coordinates, int dimensions);

	static void readSpecifiers(std::vector<int> *specifiers, int numSpecifiers);

	static bool readComplex(std::complex<double> *c);

	static bool readDouble(double *d, char endChar = ' ');

	static bool readInt(int *i, char endChar = ' ');

	static std::ofstream fout;

	static std::stringstream ssin;
};

};	//End of namespace TBTK

#endif
