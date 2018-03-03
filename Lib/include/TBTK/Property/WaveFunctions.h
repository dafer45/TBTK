/* Copyright 2016 Kristofer Björnson
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
 *  @file WaveFunctions.h
 *  @brief Property container for wave functions.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WAVE_FUNCTIONS
#define COM_DAFER45_TBTK_WAVE_FUNCTIONS

#include "TBTK/Property/AbstractProperty.h"

#include <complex>

namespace TBTK{
namespace Property{

/** @brief Property container for wave function. */
class WaveFunctions : public AbstractProperty<std::complex<double>>{
public:
	/** Constructor. */
//	WaveFunction(int dimensions, const int *ranges);

	/** Constructor. */
//	WaveFunction(int dimensions, const int *ranges, const double *data);

	/** Constructor. */
	WaveFunctions(
		const IndexTree &indexTree,
		const std::initializer_list<unsigned int> &states
	);

	/** Constructor. */
	WaveFunctions(
		const IndexTree &indexTree,
		const std::vector<unsigned int> &states
	);

	/** Constructor. */
	WaveFunctions(
		const IndexTree &indexTree,
		const std::initializer_list<unsigned int> &states,
		const std::complex<double> *data
	);

	/** Constructor. */
	WaveFunctions(
		const IndexTree &indexTree,
		const std::vector<unsigned int> &states,
		const std::complex<double> *data
	);

	/** Copy constructor. */
	WaveFunctions(const WaveFunctions &waveFunctions);

	/** Move constructor. */
	WaveFunctions(WaveFunctions &&waveFunctions);

	/** Constructor. Constructs the WaveFunctions from a serialization
	 *  string. */
	WaveFunctions(const std::string &serialization, Mode mode);

	/** Destructor. */
	~WaveFunctions();

	/** Assignment operator. */
	WaveFunctions& operator=(const WaveFunctions &rhs);

	/** Move assignment operator. */
	WaveFunctions& operator=(WaveFunctions &&rhs);

	/** Returns a vector with the state indices for which the wave function
	 *  is defined. */
	const std::vector<unsigned int>& getStates() const;

	/** Overrides AbstractProperty::operator(). */
	const std::complex<double>& operator()(
		const Index &index,
		unsigned int state
	) const;

	/** Overrides AbstractProperty::operator(). */
	std::complex<double>& operator()(
		const Index &index,
		unsigned int state
	);

	/** Get min absolute value. */
	double getMinAbs() const;

	/** Get max absolute value. */
	double getMaxAbs() const;

	/** Get min argument value. */
	double getMinArg() const;

	/** Get max argument value. */
	double getMaxArg() const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
	/** Flag indicating whether the state indices for a continuous set.
	 *  Allows for quicker access. */
	bool isContinuous;

	/** IndexTree describing*/
	std::vector<unsigned int> states;
};

inline const std::vector<unsigned int>& WaveFunctions::getStates() const{
	return states;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
