/* Copyright 2019 Kristofer Björnson
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
 *  @file TransmissionRate.h
 *  @brief Property container for transmission rates.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TRANSMISSION_RATE
#define COM_DAFER45_TBTK_TRANSMISSION_RATE

#include "TBTK/Property/EnergyResolvedProperty.h"

namespace TBTK{
namespace Property{

/** \brief Property container for transmission rates. */
class TransmissionRate : public EnergyResolvedProperty<double>{
public:
	/** Constructs a TransmissionRate.
	 *
	 *  @param energyWindow The energy window over which the Transmission
	 *  is defined. */
	TransmissionRate(const Range &energyWindow);

	/** Constructs a TransmissionRate and initializes it with data.
	 *
	 *  @param energyWindow The energy window over which the Transmission
	 *  is defined.
	 *  @param data Raw data to initialize the TransmissionRate with. */
	TransmissionRate(
		const Range &energyWindow,
		const CArray<double> &data
	);

	/** Constructor. Constructs the TransmissionRate from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the TransmissionRate.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	TransmissionRate(const std::string &serialization, Mode mode);

	/** Overrides EnergyResolvedProperty::operator+=(). */
	TransmissionRate& operator+=(const TransmissionRate &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new TransmissionRate that is the sum of this
	 *  TransmissionRate and the right hand side. */
	TransmissionRate operator+(const TransmissionRate &rhs) const;

	/** Overrides EnergyResolvedProperty::operator-=(). */
	TransmissionRate& operator-=(const TransmissionRate &rhs);

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new TransmissionRate that is the difference between this
	 *  TransmissionRate and the right hand side. */
	TransmissionRate operator-(const TransmissionRate &rhs) const;

	/** Multiplication assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side aftre the right hand side after having
	 *  been multiplied by the right hand side. */
	TransmissionRate& operator*=(const TransmissionRate &rhs);

	/** Multiplciation operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new TransmissionRate that is the product of this
	 *  TransmissionRate and the right hand side. */
	TransmissionRate operator*(const TransmissionRate &rhs) const;

	/** Overrides EnergyResolvedProperty::operator*=(). */
	TransmissionRate& operator*=(const double &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new TransmissionRate that is the product of the
	 *  TransmissionRate and the right hand side. */
	TransmissionRate operator*(const double &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side of the equation.
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new TransmissionRate that is the product of the left hand
	 *  side and the TransmissionRate. */
	friend TransmissionRate operator*(
		const double &lhs,
		const TransmissionRate &rhs
	){
		return rhs*lhs;
	}

	/** Division assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after having been divided by the right
	 *  hand side. */
	TransmissionRate& operator/=(const TransmissionRate &rhs);

	/** Division operator.
	 *
	 *  @param lhs The left hand side of the expression.
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return A new TransmissionRate that is the quotient between the
	 *  left hand side and the right hand side. */
	TransmissionRate operator/(const TransmissionRate &rhs) const;

	/** Overrides EnergyResolvedProperty::operator/=(). */
	TransmissionRate& operator/=(const double &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the equation.
	 *
	 *  @return A new TransmissionRate that is the quotient the
	 *  TransmissionRate and the right hand side. */
	TransmissionRate operator/(const double &rhs) const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
};

inline TransmissionRate& TransmissionRate::operator+=(
	const TransmissionRate &rhs
){
	EnergyResolvedProperty::operator+=(rhs);

	return *this;
}

inline TransmissionRate TransmissionRate::operator+(
	const TransmissionRate &rhs
) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate += rhs;
}

inline TransmissionRate& TransmissionRate::operator-=(
	const TransmissionRate &rhs
){
	EnergyResolvedProperty::operator-=(rhs);

	return *this;
}

inline TransmissionRate TransmissionRate::operator-(
	const TransmissionRate &rhs
) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate -= rhs;
}

inline TransmissionRate TransmissionRate::operator*(
	const TransmissionRate &rhs
) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate *= rhs;
}

inline TransmissionRate& TransmissionRate::operator*=(const double &rhs){
	EnergyResolvedProperty::operator*=(rhs);

	return *this;
}

inline TransmissionRate TransmissionRate::operator*(const double &rhs) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate *= rhs;
}

inline TransmissionRate TransmissionRate::operator/(
	const TransmissionRate &rhs
) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate /= rhs;
}

inline TransmissionRate& TransmissionRate::operator/=(const double &rhs){
	EnergyResolvedProperty::operator/=(rhs);

	return *this;
}

inline TransmissionRate TransmissionRate::operator/(const double &rhs) const{
	TransmissionRate transmissionRate = *this;

	return transmissionRate /= rhs;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
/// @endcond
