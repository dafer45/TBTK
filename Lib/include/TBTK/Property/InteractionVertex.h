/* Copyright 2018 Kristofer Björnson
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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file InteractionVertex.h
 *  @brief Property container for an InteractionVertex.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_INTERACTION_VERTEX
#define COM_DAFER45_TBTK_PROPERTY_INTERACTION_VERTEX

#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Property container for the Susceptibility. */
class InteractionVertex : public EnergyResolvedProperty<std::complex<double>>{
public:
	/** Constructs an uninitialized InteractionVertex. */
	InteractionVertex();

	/** Constructs an InteractionVertex with real energies on the Custom
	 *  format. [See AbstractProperty for detailed information about the
	 *  Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the Susceptibility should be contained.
	 *
	 *  @param energyWindow The energy window over which the
	 *  InteractionVertex is defined. */
	InteractionVertex(
		const IndexTree &indexTree,
		const Range &energyWindow
	);

	/** Constructs an InteractionVertex with real energies on the Custom
	 *  format and initializes it with data. [See AbstractProperty for
	 *  detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the InteractionVertex should be contained.
	 *
	 *  @param energyWindow The energy window over which the
	 *  InteractionVertex is defined.
	 *
	 *  @param data Raw data to initialize the InteractionVertex with. */
	InteractionVertex(
		const IndexTree &indexTree,
		const Range &energyWindow,
		const std::complex<double> *data
	);

	/** Constructs an InteractionVertex with Matsubara energies on the
	 *  Custom format. [See AbstractProperty for detailed information about
	 *  the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the InteractionVertex should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	InteractionVertex(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy
	);

	/** Constructs an InteractionVertex with Matsubara energies on the
	 *  Custom format and initializes it with data. [See AbstractProperty
	 *  for detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the InteractionVertex should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	InteractionVertex(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy,
		const std::complex<double> *data
	);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	InteractionVertex& operator+=(const InteractionVertex &rhs);

	/** Overrides EnergyResolvedProperty::operator-=(). */
	InteractionVertex& operator-=(const InteractionVertex &rhs);

	/** Overrides EnergyResolvedProperty::operator*=(). */
	InteractionVertex& operator*=(const std::complex<double> &rhs);

	/** Multiplication operator. Multiplies an Interaction vertex by a
	 *  number from the right.
	 *
	 *  @param rhs The number to multiply the InteractionVertex by.
	 *
	 *  @return A new InteractionVertex that is the product of the old
	 *  InteractionVertex and the rhs. */
	InteractionVertex operator*(const std::complex<double> &rhs) const;

	/** Multiplication operator. Multiplies an Interaction vertex by a
	 *  number from the left.
	 *
	 *  @param lhs The number to multiply the InteractionVertex by.
	 *  @param rhs The InteractionVertex to be multiplied.
	 *
	 *  @return A new InteractionVertex that is the product of the old
	 *  InteractionVertex and the lhs. */
	friend InteractionVertex operator*(
		const std::complex<double> &lhs,
		const InteractionVertex &rhs
	){
		return rhs*lhs;
	}

	/** Overrides EnergyResolvedProperty::operator/=(). */
	InteractionVertex& operator/=(const std::complex<double> &rhs);
private:
};

inline InteractionVertex& InteractionVertex::operator+=(
	const InteractionVertex &rhs
){
	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline InteractionVertex& InteractionVertex::operator-=(
	const InteractionVertex &rhs
){
	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

inline InteractionVertex& InteractionVertex::operator*=(
	const std::complex<double> &rhs
){
	EnergyResolvedProperty::operator*=(rhs);

	return *this;
}

inline InteractionVertex InteractionVertex::operator*(
	const std::complex<double> &rhs
) const{
	InteractionVertex interactionVertex = *this;

	return interactionVertex *= rhs;
}

inline InteractionVertex& InteractionVertex::operator/=(
	const std::complex<double> &rhs
){
	EnergyResolvedProperty::operator/=(rhs);

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
/// @endcond
