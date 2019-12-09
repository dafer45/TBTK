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
 *  @file SelfEnergy.h
 *  @brief Property container for the SelfEnergy.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_SELF_ENERGY
#define COM_DAFER45_TBTK_PROPERTY_SELF_ENERGY

#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Property container for the SelfEnergy. */
class SelfEnergy : public EnergyResolvedProperty<std::complex<double>>{
public:
	/** Constructs an uninitialized SelfEnergy. */
	SelfEnergy();

	/** Constructs a SelfEnergy with real energies on the Custom format.
	 *  [See AbstractProperty for detailed information about the Custom
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the SelfEnergy should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	SelfEnergy(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructs a SelfEnergy with real energies on the Custom format
	 *  and initializes it with data. [See AbstractProperty for detailed
	 *  information about the Custom format and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the SelfEnergy should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	SelfEnergy(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Constructs a SelfEnergy with Matsubara energies on the Custom
	 *  format. [See AbstractProperty for detailed information about the
	 *  Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the SelfEnergy should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	SelfEnergy(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy
	);

	/** Constructs a SelfEnergy with Matsubara energies on the Custom
	 *  format and initializes it with data. [See AbstractProperty for
	 *  detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the SelfEnergy should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	SelfEnergy(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy,
		const std::complex<double> *data
	);

	/** Constructor. Constructs the SelfEnergy from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the SelfEnergy.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	SelfEnergy(const std::string &serialization, Mode mode);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	SelfEnergy& operator+=(const SelfEnergy &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new SelfEnergy that is the sum of this SelfEnergy and the
	 *  right hand side. */
	SelfEnergy operator+(const SelfEnergy &rhs) const;

	/** Overrides EnergyResolvedProperty::operator+=(). */
	SelfEnergy& operator-=(const SelfEnergy &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new SelfEnergy that is the difference between this
	 *  SelfEnergy and the right hand side. */
	SelfEnergy operator-(const SelfEnergy &rhs) const;

	/** Overrides EnergyResolvedProperty::operator*=(). */
	SelfEnergy& operator*=(const std::complex<double> &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new SelfEnergy that is the product of the SelfEnergy and
	 *  the right hand side. */
	SelfEnergy operator*(const std::complex<double> &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side of the equation.
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new SelfEnergy that is the product of the left and right
	 *  hand side. */
	friend SelfEnergy operator*(
		const std::complex<double> &lhs,
		const SelfEnergy &rhs
	){
		return rhs*lhs;
	}

	/** Overrides EnergyResolvedProperty::operator/=(). */
	SelfEnergy& operator/=(const std::complex<double> &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new SelfEnergy that is the quotient between this
	 *  SelfEnergy and the right hand side. */
	SelfEnergy operator/(const std::complex<double> &rhs) const;

	/** Overrides EnergyResolvedProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
};

inline SelfEnergy& SelfEnergy::operator+=(const SelfEnergy &rhs){
	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline SelfEnergy SelfEnergy::operator+(const SelfEnergy &rhs) const{
	SelfEnergy selfEnergy = *this;

	return selfEnergy += rhs;
}

inline SelfEnergy& SelfEnergy::operator-=(const SelfEnergy &rhs){
	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

inline SelfEnergy SelfEnergy::operator-(const SelfEnergy &rhs) const{
	SelfEnergy selfEnergy = *this;

	return selfEnergy -= rhs;
}

inline SelfEnergy& SelfEnergy::operator*=(const std::complex<double> &rhs){
	EnergyResolvedProperty::operator*=(rhs);

	return *this;
}

inline SelfEnergy SelfEnergy::operator*(
	const std::complex<double> &rhs
) const{
	SelfEnergy selfEnergy = *this;

	return selfEnergy *= rhs;
}

inline SelfEnergy& SelfEnergy::operator/=(const std::complex<double> &rhs){
	EnergyResolvedProperty::operator/=(rhs);

	return *this;
}

inline SelfEnergy SelfEnergy::operator/(const std::complex<double> &rhs) const{
	SelfEnergy selfEnergy = *this;

	return selfEnergy /= rhs;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
/// @endcond
