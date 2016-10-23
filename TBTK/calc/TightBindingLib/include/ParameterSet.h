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

/** @package TBTKcalc ParameterSet
 *  @file ParameterSet.h
 *  @brief Set of parameters
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
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

	/** Add string parameter. */
	void addString(std::string name, std::string value);

	/** Add boolean parameter. */
	void addBool(std::string name, bool value);

	/** Get integer parameter. */
	int getInt(std::string name) const;

	/** Get double parameter. */
	double getDouble(std::string name) const;

	/** Get complex parameter. */
	std::complex<double> getComplex(std::string name) const;

	/** Get string parameter. */
	std::string getString(std::string name) const;

    /** Get boolean parameter. */
	bool getBool(std::string name) const;

	/** Get number of integer parameters. */
	int getNumInt() const;

	/** Get number of double parameters. */
	int getNumDouble() const;

	/** Get number of complex parameters. */
	int getNumComplex() const;

	/** Get number of string parameters. */
	int getNumString() const;

	/** Get number of boolean parameters. */
	int getNumBool() const;

	/** Get integer name. */
	std::string getIntName(int n) const;

	/** Get double name. */
	std::string getDoubleName(int n) const;

	/** Get complex name. */
	std::string getComplexName(int n) const;

	/** Get string name. */
	std::string getStringName(int n) const;

	/** Get booleanname. */
	std::string getBoolName(int n) const;

	/** Get integer value. */
	int getIntValue(int n) const;

	/** Get double value. */
	double getDoubleValue(int n) const;

	/** Get complex value. */
	std::complex<double> getComplexValue(int n) const;

	/** Get string value. */
	std::string getStringValue(int n) const;

	/** Get boolean value. */
	bool getBoolValue(int n) const;

	/** Returns true if an integer parameter with given name exists. */
	bool intExists(std::string name) const;

	/** Returns true if a double parameter with given name exists. */
	bool doubleExists(std::string name) const;

	/** Returns true if a complex parameter with given name exists. */
	bool complexExists(std::string name) const;

    /** Returns true if an string parameter with given name exists. */
	bool stringExists(std::string name) const;

    /** Returns true if an boolean parameter with given name exists. */
	bool boolExists(std::string name) const;
private:
	/** Integer parameters. */
	std::vector<std::tuple<std::string, int>> intParams;

	/** Double parameters. */
	std::vector<std::tuple<std::string, double>> doubleParams;

	/** Complex parameters. */
	std::vector<std::tuple<std::string, std::complex<double>>> complexParams;

    /** String parameters. */
	std::vector<std::tuple<std::string, std::string>> stringParams;

    /** Boolean parameters. */
	std::vector<std::tuple<std::string, bool>> boolParams;
};

};	//End of namespace Util
};	//End of namespace TBTK

#endif
