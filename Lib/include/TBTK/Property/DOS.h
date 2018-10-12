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
 *  @file DOS.h
 *  @brief Property container for density of states (DOS).
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS
#define COM_DAFER45_TBTK_DOS

#include "TBTK/Property/EnergyResolvedProperty.h"

namespace TBTK{
namespace Property{

/** \brief Property container for density of states (DOS). */
class DOS : public EnergyResolvedProperty<double>{
public:
	/** Constructs a Density.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to us for the energy. */
	DOS(double lowerBound, double upperBound, int resolution);

	/** Constructs a Density and initializes it with data.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to us for the energy.
	 *  @param data Raw data to initialize the DOS with. */
	DOS(
		double lowerBound,
		double upperBound,
		int resolution,
		const double *data
	);

	/** Constructor. Constructs the DOS from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the DOS.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	DOS(const std::string &serialization, Mode mode);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	DOS& operator+=(const DOS &rhs);

	/** Overrides EnergyResolvedProperty::operator-=(). */
	DOS& operator-=(const DOS &rhs);

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
};

inline DOS& DOS::operator+=(const DOS &rhs){
	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline DOS& DOS::operator-=(const DOS &rhs){
	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
