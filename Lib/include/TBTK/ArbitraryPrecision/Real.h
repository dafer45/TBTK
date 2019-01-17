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
 *  @file Real.h
 *  @brief Arbitrary precision real number.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ARBITRARY_PRECISION_REAL
#define COM_DAFER45_TBTK_ARBITRARY_PRECISION_REAL

#include <string>

#include <gmpxx.h>

namespace TBTK{
namespace ArbitraryPrecision{

class Real{
public:
	/** Constructor. Constructs an uninitialized Real number. An already
	 *  initialized Real number has to be assigned before the number is
	 *  ready to be use. In particular, assigning a double or a string to
	 *  an uninitialized Real number result in undefined behavior. */
	Real();

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store the number. */
	Real(unsigned int precision);

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store the number. */
	Real(unsigned int precision, double value);

	/** Constructor.
	 *
	 *  @param precision The number of bits used to store the number. */
	Real(unsigned int precision, const std::string &value);

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after assignment. */
	const Real& operator=(double rhs);

	/** Assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the assignment. */
	const Real& operator=(const std::string &rhs);

	/** Addition assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the assignment has occured. */
	Real& operator+=(const Real &rhs);

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the addition. */
	Real operator+(const Real &rhs) const;

	/** Subtraction assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the assignment has occured. */
	Real& operator-=(const Real &rhs);

	/** Subtraction operator.
	 *
	 *  @param The right hand side of the expression.
	 *
	 *  @return The result of the subtraction. */
	Real operator-(const Real &rhs) const;

	/** Multiplication assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the assignment has occured. */
	Real& operator*=(const Real &rhs);

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the multiplication. */
	Real operator*(const Real &rhs) const;

	/** Division assignment operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The left hand side after the assignment has occured. */
	Real& operator/=(const Real &rhs);

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the division. */
	Real operator/(const Real &rhs) const;

	/** ostream operator.
	 *
	 *  @param os The ostream to write to.
	 *  @param real The Real number to write. */
	friend std::ostream& operator<<(std::ostream &os, const Real &real);

	/** Get the value on the GMP format mpf_t.
	 *
	 *  @return The value as stored on mpf_t type. */
	const mpf_t& getValue() const;

	/** Get the value as a double.
	 *
	 *  @return The value truncated to double precision. */
	double getDouble() const;

	/** Get the precision.
	 *
	 *  @return The number of bits used to store the number. */
	mp_bitcnt_t getPrecision() const;
private:
	/** The value. */
	mpf_t value;
};

inline Real::Real(){
}

inline Real::Real(unsigned int precision){
	mpf_init2(value, precision);
}

inline Real::Real(unsigned int precision, double value){
	mpf_init2(this->value, precision);
	operator=(value);
}

inline Real::Real(unsigned int precision, const std::string &value){
	mpf_init2(this->value, precision);
	operator=(value);
}

inline const Real& Real::operator=(double rhs){
	mpf_set_d(value, rhs);

	return *this;
}

inline const Real& Real::operator=(const std::string &rhs){
	mpf_set_str(value, rhs.c_str(), 10);

	return *this;
}

inline Real& Real::operator+=(const Real &rhs){
	mpf_add(value, value, rhs.value);

	return *this;
}

inline Real Real::operator+(const Real &rhs) const{
	Real real = *this;

	return real += rhs;
}

inline Real& Real::operator-=(const Real &rhs){
	mpf_sub(value, value, rhs.value);

	return *this;
}

inline Real Real::operator-(const Real &rhs) const{
	Real real = *this;

	return real -= rhs;
}

inline Real& Real::operator*=(const Real &rhs){
	mpf_mul(value, value, rhs.value);

	return *this;
}

inline Real Real::operator*(const Real &rhs) const{
	Real real = *this;

	return real *= rhs;
}

inline Real& Real::operator/=(const Real &rhs){
	mpf_div(value, value, rhs.value);

	return *this;
}

inline Real Real::operator/(const Real &rhs) const{
	Real real = *this;

	return real /= rhs;
}

inline std::ostream& operator<<(std::ostream &os, const Real &real){
	os << real.value;

	return os;
}

inline const mpf_t& Real::getValue() const{
	return value;
}

inline double Real::getDouble() const{
	return mpf_get_d(value);
}

inline mp_bitcnt_t Real::getPrecision() const{
	return mpf_get_prec(value);
}

}; //End of namespace ArbitraryPrecision
}; //End of namesapce TBTK

#endif
