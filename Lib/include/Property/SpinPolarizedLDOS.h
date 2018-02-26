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
 *  @file SpinPolarizedLDOS.h
 *  @brief Property container for spin-polarized local density of states
 *    (spin-polarized LDOS).
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SPIN_POLARIZED_LDOS
#define COM_DAFER45_TBTK_SPIN_POLARIZED_LDOS

#include "AbstractProperty.h"
#include "SpinMatrix.h"

#include <complex>

namespace TBTK{
namespace Property{

/** @brief Property container for spin-polarized local density of states
 *    (spin-polarized LDOS). */
class SpinPolarizedLDOS : public AbstractProperty<SpinMatrix>{
public:
	/** Constructor. */
	SpinPolarizedLDOS(
		int dimensions,
		const int *ranges,
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Constructor. */
	SpinPolarizedLDOS(
		int dimensions,
		const int *ranges,
		double lowerBound,
		double upperBound,
		int resolution,
		const SpinMatrix *data
	);

	/** Constructor. */
	SpinPolarizedLDOS(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Constructor. */
	SpinPolarizedLDOS(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		int resolution,
		const SpinMatrix *data
	);

	/** Copy constructor. */
	SpinPolarizedLDOS(const SpinPolarizedLDOS &spinPolarizedLDOS);

	/** Move constructor. */
	SpinPolarizedLDOS(SpinPolarizedLDOS &&spinPolarizedLDOS);

	/** Constructor. Construct the SpinPolarizedLDOS from a serialization
	 *  string. */
	SpinPolarizedLDOS(const std::string &serialization, Mode mode);

	/** Destructor. */
	~SpinPolarizedLDOS();

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Assignment operator. */
	SpinPolarizedLDOS& operator=(const SpinPolarizedLDOS &rhs);

	/** Move assignment operator. */
	SpinPolarizedLDOS& operator=(SpinPolarizedLDOS &&rhs);

	/** Overrides AbstractProperty::serialize(). */
	std::string serialize(Mode mode) const;
private:
	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	int resolution;
};

inline double SpinPolarizedLDOS::getLowerBound() const{
	return lowerBound;
}

inline double SpinPolarizedLDOS::getUpperBound() const{
	return upperBound;
}

inline int SpinPolarizedLDOS::getResolution() const{
	return resolution;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
