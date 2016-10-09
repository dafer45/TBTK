/** @package TBTKcalc ParameterSet
 *  @file ParameterSet.h
 *  @brief Set of parameters
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARAMETER_SET
#define COM_DAFER45_TBTK_PARAMETER_SET

#include <vector>
#include <tuple>
#include <string>
#include <complex>

namespace TBTK{
namespace Util{

class ParameterSet{
public:
	/** Constructor. */
	ParameterSet();

	/** Destrutor. */
	~ParameterSet();

	/** Add integer parameter. */
	void addInt(std::string name, int value);

	/** Add double parameter. */
	void addDouble(std::string name, double value);

	/** Add complex parameter. */
	void addComplex(std::string name, std::complex<double> value);

	/** Get integer parameter. */
	int getInt(std::string name);

	/** Get double parameter. */
	double getDouble(std::string name);

	/** Get complex parameter. */
	std::complex<double> getComplex(std::string name);

	/** Get number of integer parameters. */
	int getNumInt();

	/** Get number of double parameters. */
	int getNumDouble();

	/** Get number of complex parameters. */
	int getNumComplex();

	/** Get integer name. */
	std::string getIntName(int n);

	/** Get double name. */
	std::string getDoubleName(int n);

	/** Get complex name. */
	std::string getComplexName(int n);

	/** Get integer value. */
	int getIntValue(int n);

	/** Get double value. */
	double getDoubleValue(int n);

	/** Get complex value. */
	std::complex<double> getComplexValue(int n);
private:
	/** Integer parameters. */
	std::vector<std::tuple<std::string, int>> intParams;

	/** Double parameters. */
	std::vector<std::tuple<std::string, double>> doubleParams;

	/** Complex parameters. */
	std::vector<std::tuple<std::string, std::complex<double>>> complexParams;
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif
