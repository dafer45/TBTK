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

/** @package TBTKcalc
 *  @file GreensFunction.h
 *  @brief Property container for the Green's function.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_GREENS_FUNCTION
#define COM_DAFER45_TBTK_PROPERTY_GREENS_FUNCTION

#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Property container for the Green's function.
 *
 *  The GreensFunction is an EnergyResolvedProperty with DataType
 *  std::complex<double>.
 *
 *  # Example
 *  \snippet Property/GreensFunction.cpp GreensFunction
 *  ## Output
 *  \snippet output/Property/GreensFunction.output GreensFunction */
class GreensFunction : public EnergyResolvedProperty<std::complex<double>>{
public:
	/** Enum class for specifying the Green's function type. */
	enum class Type{
		TimeOrdered,
		Advanced,
		Retarded,
		Principal,
		NonPrincipal,
		Matsubara
	};

	/** Constructs an uninitialized GreensFunction. */
	GreensFunction();

	/** Constructs a GreensFunction with real energies on the Custom
	 *  format. [See AbstractProperty for detailed information about the
	 *  Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	GreensFunction(
		const IndexTree &indexTree,
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructs a GreensFunction with real energies on the Custom format
	 *  and initializes it with data. [See AbstractProperty for detailed
	 *  information about the Custom format and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	GreensFunction(
		const IndexTree &indexTree,
		Type type,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const std::complex<double> *data
	);

	/** Constructs a GreensFunction with Matsubara energies on the Custom
	 *  format. [See AbstractProperty for detailed information about the
	 *  Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	GreensFunction(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy
	);

	/** Constructs a GreensFunction with Matsubara energies on the Custom
	 *  format and initializes it with data. [See AbstractProperty for
	 *  detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the GreensFunction should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the GreensFunction with. */
	GreensFunction(
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex,
		double fundamentalMatsubaraEnergy,
		const std::complex<double> *data
	);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	GreensFunction& operator+=(const GreensFunction &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new GreensFunction that is the sum of this GreensFunction
	 *  and the right hand side. */
	GreensFunction operator+(const GreensFunction &rhs) const;

	/** Overrides EnergyResolvedProperty::operator-=(). */
	GreensFunction& operator-=(const GreensFunction &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new GreensFunction that is the difference between this
	 *  GreensFunction and the right hand side. */
	GreensFunction operator-(const GreensFunction &rhs) const;

	/** Overrides EnergyResolvedProperty::operator*=(). */
	GreensFunction& operator*=(const std::complex<double> &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new GreensFunction that is the product of the
	 *  GreensFunction and the right hand side. */
	GreensFunction operator*(const std::complex<double> &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side of the equation.
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new GreensFunction that is the product of the left hand
	 *  side and the GreensFunction. */
	friend GreensFunction operator*(
		const std::complex<double> &lhs,
		const GreensFunction &rhs
	){
		return rhs*lhs;
	}

	/** Overrides EnergyResolvedProperty::operator/=(). */
	GreensFunction& operator/=(const std::complex<double> &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new GreensFunction that is the quotient between the
	 *  GreensFunction and the right hand side. */
	GreensFunction operator/(const std::complex<double> &rhs) const;

	/** Get the Green's function type.
	 *
	 *  @return The Green's function type. */
	Type getType() const;

	/** Implements Streamable::toString(). */
	std::string toString() const;
private:
	/** The Green's function type. */
	Type type;
};

inline GreensFunction& GreensFunction::operator+=(const GreensFunction &rhs){
	TBTKAssert(
		type == rhs.type,
		"GreensFunction::operator+=()",
		"Incompatible Green's function types.",
		""
	);

	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline GreensFunction GreensFunction::operator+(const GreensFunction &rhs) const{
	GreensFunction greensFunction = *this;

	return greensFunction += rhs;
}

inline GreensFunction& GreensFunction::operator-=(const GreensFunction &rhs){
	TBTKAssert(
		type == rhs.type,
		"GreensFunction::operator-=()",
		"Incompatible Green's function types.",
		""
	);

	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

inline GreensFunction GreensFunction::operator-(const GreensFunction &rhs) const{
	GreensFunction greensFunction = *this;

	return greensFunction -= rhs;
}

inline GreensFunction& GreensFunction::operator*=(const std::complex<double> &rhs){
	EnergyResolvedProperty::operator*=(rhs);

	return *this;
}

inline GreensFunction GreensFunction::operator*(const std::complex<double> &rhs) const{
	GreensFunction greensFunction = *this;

	return greensFunction *= rhs;
}

inline GreensFunction& GreensFunction::operator/=(const std::complex<double> &rhs){
	EnergyResolvedProperty::operator/=(rhs);

	return *this;
}

inline GreensFunction GreensFunction::operator/(const std::complex<double> &rhs) const{
	GreensFunction greensFunction = *this;

	return greensFunction /= rhs;
}

inline GreensFunction::Type GreensFunction::getType() const{
	return type;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
