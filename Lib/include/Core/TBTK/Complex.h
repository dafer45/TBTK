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

#include "TBTK/Boolean.h"
#include "TBTK/PseudoSerializable.h"
#include "TBTK/Real.h"

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

	/** Constructor.
	 *
	 *  @param value The value to initilize the Complex with. */
	constexpr Complex(Real real, Real imag) : value(real, imag) {}

	/** Constructor.
	 *
	 *  @param value The value to initilize the Complex with. */
	constexpr Complex(double value) : value(value) {}

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

	/** Addition assignment operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The left hand side after the addition has occured. */
	friend std::complex<double>& operator+=(
		std::complex<double> &lhs,
		const Complex &rhs
	){
		lhs += rhs.value;

		return lhs;
	}

	/** Addition operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the addition. */
	friend Complex operator+(const Complex &lhs, const double &rhs){
		Complex result(lhs);
		result.value += rhs;

		return result;
	}

	/** Addition operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the addition. */
	friend Complex operator+(const double &lhs, const Complex &rhs){
		Complex result(rhs);
		result.value += lhs;

		return result;
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

	/** Subtraction operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the subtraction. */
	Complex operator-(const Complex &rhs) const{
		Complex result(*this);
		result -= rhs;

		return result;
	}

	/** Subtraction operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the subtraction. */
	friend Complex operator-(const Complex &lhs, const double &rhs){
		Complex result(lhs);
		result.value -= rhs;

		return result;
	}

	/** Subtraction operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the subtraction. */
	friend Complex operator-(const double &lhs, const Complex &rhs){
		Complex result(lhs - rhs.value.real(), -rhs.value.imag());

		return result;
	}

	/** Unary minus operator.
	 *
	 *  @return The negative of the Complex number. */
	Complex operator-() const{
		return Complex(-value);
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

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the multiplication. */
	friend Complex operator*(const Complex &lhs, const double &rhs){
		Complex result(lhs);
		result.value *= rhs;

		return result;
	}

	/** Multiplication operator.
	 *
	 *  @param lhs The left hand side.
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the multiplication. */
	friend Complex operator*(const double &lhs, const Complex &rhs){
		Complex result(rhs);
		result.value *= lhs;

		return result;
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

	/** Division operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return The result of the division. */
	friend Complex operator/(const Complex &lhs, const double &rhs){
		Complex result(lhs);
		result.value /= rhs;

		return result;
	}

	/** Comparison operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return True if the two Complex numbers are equal, otherwise false. */
	Boolean operator==(const Complex &rhs) const{
		return value == rhs.value;
	}

	/** Not equal operator.
	 *
	 *  @param rhs The right hand side.
	 *
	 *  @return False if the two Complex numbers are equal, otherwise true. */
	Boolean operator!=(const Complex &rhs) const{
		return value != rhs.value;
	}

	/** ostream operator.
	 *
	 *  @param os The ostream to write to.
	 *  @param complex The Complex to write.
	 *
	 *  @return The ostream. */
	friend std::ostream& operator<<(std::ostream &os, const Complex &complex){
		os << complex.value;

		return os;
	}

	/** istream operator.
	 *
	 *  @param is The istream to read from.
	 *  @param complex The Complex to read to.
	 *
	 *  @return The istream. */
	friend std::istream& operator>>(std::istream &is, Complex &complex){
		is >> complex.value;

		return is;
	}

	/** Serialize Complex. Note that Complex is PseudoSerializable rather
	 *  than Serializable. This means that the Serializable interface is
	 *  implemented non-virtually.
	 *
	 *  @param mode Serialization mode.
	 *
	 *  @return Serialized string representation of the Complex. */
	std::string serialize(Serializable::Mode mode) const;

	/** Get the real part.
	 *
	 *  @param complex The Complex number.
	 *
	 *  @return The real part. */
	friend constexpr Real real(const Complex &complex){
		return real(complex.value);
	}

	/** Get the imaginary part.
	 *
	 *  @param complex The Complex number.
	 *
	 *  @return The imaginary part. */
	friend constexpr Real imag(const Complex &complex){
		return imag(complex.value);
	}

	/** Complex conjugate.
	 *
	 *  @param complex The Complex number.
	 *
	 *  @return The complex conjugate. */
	friend Complex conj(const Complex &complex){
		return conj(complex.value);
	}

	/** Absolute value.
	 *
	 *  @param complex The Complex number.
	 *
	 *  @return The absolute value. */
	friend Real abs(const Complex &complex){
		return abs(complex.value);
	}
private:
	/** Value. */
	std::complex<double> value;
};

inline Complex::Complex(
	const std::string &serialization,
	Serializable::Mode mode
){
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
