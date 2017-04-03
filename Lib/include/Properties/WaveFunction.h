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
 *  @file WaveFunction.h
 *  @brief Property container for wave function
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WAVE_FUNCTION
#define COM_DAFER45_TBTK_WAVE_FUNCTION

#include "AbstractProperty.h"

#include <complex>

namespace TBTK{
namespace Property{

/** Container for density. */
class WaveFunction : public AbstractProperty<std::complex<double>>{
public:
	/** Constructor. */
//	WaveFunction(int dimensions, const int *ranges);

	/** Constructor. */
//	WaveFunction(int dimensions, const int *ranges, const double *data);

	/** Constructor. */
	WaveFunction(
		const IndexTree &indexTree,
		const std::initializer_list<unsigned int> &states
	);

	/** Constructor. */
	WaveFunction(
		const IndexTree &indexTree,
		const std::vector<unsigned int> &states
	);

	/** Constructor. */
	WaveFunction(
		const IndexTree &indexTree,
		const std::initializer_list<unsigned int> &states,
		const std::complex<double> *data
	);

	/** Constructor. */
	WaveFunction(
		const IndexTree &indexTree,
		const std::vector<unsigned int> &states,
		const std::complex<double> *data
	);

	/** Copy constructor. */
	WaveFunction(const WaveFunction &waveFunction);

	/** Move constructor. */
	WaveFunction(WaveFunction &&waveFunction);

	/** Destructor. */
	~WaveFunction();

	/** Assignment operator. */
	WaveFunction& operator=(const WaveFunction &rhs);

	/** Move assignment operator. */
	WaveFunction& operator=(WaveFunction &&rhs);

	/** Returns a vector with the state indices for which the wave function
	 *  is defined. */
	const std::vector<unsigned int>& getStates() const;

	/** Overrides AbstractProperty::operator(). */
	std::complex<double> operator()(
		const Index &index,
		unsigned int state
	) const;
private:
	/** Flag indicating whether the state indices for a continuous set.
	 *  Allows for quicker access. */
	bool isContinuous;

	/** IndexTree describing*/
	std::vector<unsigned int> states;
};

inline const std::vector<unsigned int>& WaveFunction::getStates() const{
	return states;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
