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
 *  @brief Arbitrary precision complex number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARBITRARY_PRECISION_COMPLEX
#define COM_DAFER45_TBTK_ARBITRARY_PRECISION_COMPLEX

#include "TBTK/ArbitraryPrecision/Real.h"
#include "TBTK/Streams.h"

#include <complex>
#include <string>

#include <gmpxx.h>

namespace TBTK{
namespace ArbitraryPrecision{

class Complex{
public:
	/** Constructor. Constructs an uninitialized Complex number. An already
	 *  initialized Complex number has to be assigned before the number is
	 *  ready to be used. In particular, asigning a complex<double> or a
	 *  string to an uninitialized Real number result in undefined
	 *  behavior. */
	Complex();

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store each of the real
	 *  and imaginary components. */
	Complex(unsigned int precision);

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store each of the real
	 *  and imaginary components.
	 *
	 *  @param value The value. */
	Complex(unsigned int precision, const std::complex<double> &value);

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store each of the real
	 *  and imaginary components.
	 *
	 *  @param real The real component.
	 *  @param imag The imaginary component. */
	Complex(unsigned int precision, double real, double imag);

	/** Constructor.
	 *
	 *  @param real The real component.
	 *  @param The imaginary component. */
	Complex(const Real &real, const Real &imag);

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after assignment. */
	const Complex& operator=(const std::complex<double> &rhs);

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression. Must be formated
	 *  as "REAL + iIMAG", where REAL and IMAG are the real and imaginary
	 *  components, respectively.
	 *
	 *  @return The left hand side after assignment. */
	const Complex& operator=(const std::string &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the addition. */
	const Complex operator+(const Complex &rhs) const;

	/** Subtraction operator.
	 *
	 *  @param The right hand side of the expression.
	 *
	 *  @return The result of the subtraction. */
	const Complex operator-(const Complex &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the multiplication. */
	const Complex operator*(const Complex &rhs) const;

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the division. */
	const Complex operator/(const Complex &rhs) const;

	/** Get the real component.
	 *
	 *  @return The real component. */
	const Real& getReal() const;

	/** Get the imaginary component.
	 *
	 *  @return The imaginary component. */
	const Real& getImag() const;
private:
	/** The real component. */
	Real real;

	/** The imaginary component. */
	Real imag;
};

inline Complex::Complex(){
}

inline Complex::Complex(
	unsigned int precision
):
	real(precision),
	imag(precision)
{
}

inline Complex::Complex(
	unsigned int precision,
	const std::complex<double> &value
) :
	real(precision, value.real()),
	imag(precision, value.imag())
{
}

inline Complex::Complex(
	unsigned int precision,
	double real,
	double imag
) :
	real(precision, real),
	imag(precision, imag)
{
}

inline Complex::Complex(
	const Real &real,
	const Real &imag
) :
	real(real),
	imag(imag)
{
}

inline const Complex& Complex::operator=(const std::complex<double> &rhs){
	real = rhs.real();
	imag = rhs.imag();

	return *this;
}

inline const Complex& Complex::operator=(const std::string &rhs){
	real = rhs.substr(0, rhs.find(" +"));
	imag = rhs.substr(rhs.find("i") + 1, rhs.size());

	return *this;
}

inline const Complex Complex::operator+(const Complex &rhs) const{
	return Complex(real + rhs.real, imag + rhs.imag);
}

inline const Complex Complex::operator-(const Complex &rhs) const{
	return Complex(real - rhs.real, imag - rhs.imag);
}

inline const Complex Complex::operator*(const Complex &rhs) const{
	return Complex(
		real*rhs.real - imag*rhs.imag,
		real*rhs.imag + imag*rhs.real
	);
}

inline const Complex Complex::operator/(const Complex &rhs) const{
	Real denominator = rhs.real*rhs.real + rhs.imag*rhs.imag;

	return Complex(
		(real*rhs.real + imag*rhs.imag)/denominator,
		(imag*rhs.real - real*rhs.imag)/denominator
	);
}

inline const Real& Complex::getReal() const{
	return real;
}

inline const Real& Complex::getImag() const{
	return imag;
}

//Complex pow(Complex base, long unsigned int power);
Complex pow(Complex base, long unsigned int power){
	Complex result;
	if(power%2)
		result = base;
	else{
		mp_bitcnt_t realPrecision = base.getReal().getPrecision();
		mp_bitcnt_t imagPrecision = base.getReal().getPrecision();
		if(realPrecision > imagPrecision)
			result = Complex(realPrecision, 1);
		else
			result = Complex(realPrecision, 1);
	}

	while(power >>= 1){
		//Update to *= when this has been implemented.
		base = base*base;
		if(power%2){
			//Update to *= when this has been implemented.
			result = result*base;
		}
	}

	return result;
}

inline Complex pow(const Complex &base, long int power){
	if(power < 0){
		Complex result = pow(base, (long unsigned int)(-power));

		return Complex(result.getReal().getPrecision(), 1)/result;
	}
	else{
		return pow(base, (long unsigned int)power);
	}
}

inline Complex pow(const Complex &base, int power){
	return pow(base, (long int)power);
}

}; //End of namespace ArbitraryPrecision
}; //End of namesapce TBTK

#endif
