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

/** @package TBTKcalc
 *  @file Complex.h
 *  @brief Complex number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_COMPLEX
#define COM_DAFER45_TBTK_COMPLEX

#include "TBTK/PseudoSerializable.h"

#include <complex>
#include <sstream>

namespace TBTK{

#if TBTK_WRAP_PRIMITIVE_TYPES

/** @brief Complex number. */
class Complex : PseudoSerializable{
public:
	/** Constructor. */
	Complex(){};

	/** Constructor.
	 *
	 *  @param value The value to initilize the Complex with. */
	constexpr Complex(std::complex<double> value) : value(value) {}

	/** Constructs an Index from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Complex.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Complex(const std::string &serialization, Serializable::Mode mode);

	/** Type conversion operator. */
	constexpr operator std::complex<double>() const{
		return value;
	};

	/** Assignment operator.
	 *
	 *  @param value The value to assign the Complex.
	 *
	 *  @return The Complex after assignment has occured. */
	Complex& operator=(const std::complex<double> &rhs){
		value = rhs;

		return *this;
	}

	/** Addition assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Complex after the addition has occured. */
	Complex& operator+=(const Complex &rhs){
		value += rhs.value;

		return *this;
	}

	/** Subtraction assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Complex after the subtraction has occured. */
	Complex& operator-=(const Complex &rhs){
		value -= rhs.value;

		return *this;
	}

	/** Multiplication assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Complex after the multiplication has occured. */
	Complex& operator*=(const Complex &rhs){
		value *= rhs.value;

		return *this;
	}

	/** Division assignment operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The Complex after the division has occured. */
	Complex& operator/=(const Complex &rhs){
		value /= rhs.value;

		return *this;
	}

	/** Serialize Complex. Note that Complex is PseudoSerializable rather
	 *  than Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Complex. */
	std::string serialize(Serializable::Mode mode) const;
private:
	/** Value. */
	std::complex<double> value;
};

inline Complex::Complex(const std::string &serialization, Serializable::Mode mode){
	switch(mode){
	case Serializable::Mode::JSON:
	{
		std::stringstream ss(serialization);
		ss >> value;

		break;
	}
	default:
		TBTKExit(
			"Complex::Complex()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

inline std::string Complex::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		std::stringstream ss;
		ss << value;

		return ss.str();
	}
	default:
		TBTKExit(
			"Complex::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

#else
	typedef std::complex<double> Complex;
#endif

};	//End of namespace TBTK

#endif
