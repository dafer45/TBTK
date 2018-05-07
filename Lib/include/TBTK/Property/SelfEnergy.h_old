/* Copyright 2017 Kristofer Björnson
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
 *  @file SelfEnergy.h
 *  @brief Property container for self-energy.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SELF_ENERGY
#define COM_DAFER45_TBTK_SELF_ENERGY

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Property container for self-energy. */
class SelfEnergy : public AbstractProperty<std::complex<double>>{
public:
	/** Constructor. */
	SelfEnergy();

	/** Constructor. */
	SelfEnergy(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructor. */
	SelfEnergy(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Copy constructor. */
	SelfEnergy(const SelfEnergy &selfEnergy);

	/** Move constructor. */
	SelfEnergy(SelfEnergy &&selfEnergy);

	/** Destructor. */
	~SelfEnergy();

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution (number of energy intervals). */
	unsigned int getResolution() const;

	/** Assignment operator. */
	const SelfEnergy& operator=(const SelfEnergy &rhs);

	/** Move assignment operator. */
	const SelfEnergy& operator=(SelfEnergy &&rhs);
private:
	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals) */
	unsigned int resolution;
};

inline double SelfEnergy::getLowerBound() const{
	return lowerBound;
}

inline double SelfEnergy::getUpperBound() const{
	return upperBound;
}

inline unsigned int SelfEnergy::getResolution() const{
	return resolution;
}

inline const SelfEnergy& SelfEnergy::operator=(
	const SelfEnergy &rhs
){
	if(this != &rhs){
		AbstractProperty::operator=(rhs);

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

inline const SelfEnergy& SelfEnergy::operator=(
	SelfEnergy &&rhs
){
	if(this != &rhs){
		AbstractProperty::operator=(std::move(rhs));

		lowerBound = rhs.lowerBound;
		upperBound = rhs.upperBound;
		resolution = rhs.resolution;
	}

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
