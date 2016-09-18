/** @package TBTKcalc
 *  @file FileParser.h
 *  @brief Reads and writes Model from and to text files.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FILE_PARSER
#define COM_DAFER45_TBTK_FILE_PARSER

#include "Model.h"

#include <string>
#include <fstream>
#include <sstream>

namespace TBTK{

/** Reads and write
 */
class FileParser{
public:
	/** Enum for indicating storage mode. */
	enum class AmplitudeMode{ALL, ALL_EXCEPT_HC};

	/** Write Model to file. */
	static void writeModel(
		Model *model,
		std::string fileName,
		AmplitudeMode amplitudeMode,
		std::string description
	);

	/** Read Model from file. */
	static Model* readModel(std::string fileName);
private:
	/** Open output stream. */
	static void openOutput(std::string fileName);

	/** Close output stream. */
	static void closeOutput();

	/** Read input strem into internal input stream buffer (ssin). */
	static void readInput(std::string fileName);

	/** Write line breaks. */
	static void writeLineBreaks(int numLineBreaks);

	/** Write tabs. */
	static void writeTabs(int numTabs);

	/** Write a complex<double>. */
	static void write(std::complex<double> value);

	/** Write an Index. Example: [x y s]. */
	static void write(const Index &index);

	/** Write coordinates. Example: (0.1, 0.2, 0.3). */
	static void writeCoordinates(const double *coordinates, int numCoordinates);

	/** Write specifiers. Example: <1 3>. */
	static void writeSpecifiers(const int *specifiers, int numSpecifiers);

	/** Write description comment. */
	static void writeDescription(std::string description);

	/** Write HoppingAmplitudes. */
	static void writeAmplitudes(Model *model, AmplitudeMode amplitudeMode);

	/** Write Geometry. */
	static void writeGeometry(Model *model);

	/** Reomve comments from file. */
	static void removeComments();

	/** Remove initial whitespaces. */
	static void removeInitialWhiteSpaces();

	/** Read a parameter */
	static int readParameter(std::string parameterName, std::string parentStructure);

	/** Read HoppingAmplitudes. */
	static void readAmplitudes(Model *model);

	/** Read Geometry. */
	static void readGeometry(Model *model);

	/** Read one HoppingAmplitude. */
	static HoppingAmplitude* readHoppingAmplitude();

	/** Read an Index. */
	static Index* readIndex();

	/** Read coordinates. Example (0.1, 0.2, 0.3). */
	static void readCoordinates(std::vector<double> *coordinates, int dimensions);

	/** Read specifiers. Example: <1 3>. */
	static void readSpecifiers(std::vector<int> *specifiers, int numSpecifiers);

	/** Read a complex<double>. */
	static bool readComplex(std::complex<double> *c);

	/** Read a double. */
	static bool readDouble(double *d, char endChar = ' ');

	/** Read an int. */
	static bool readInt(int *i, char endChar = ' ');

	/** Output file stream for writing. */
	static std::ofstream fout;

	/** String stream for parsing input. */
	static std::stringstream ssin;
};

};	//End of namespace TBTK

#endif
