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
	/** Constructor.
	 *
	 *  @param precision The number of bits used to store the number. */
	Real(unsigned int precision);

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

	/** Addition operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the addition. */
	const Real operator+(const Real &rhs) const;

	/** Subtraction operator.
	 *
	 *  @param The right hand side of the expression.
	 *
	 *  @return The result of the subtraction. */
	const Real operator-(const Real &rhs) const;

	/** Multiplication operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the multiplication. */
	const Real operator*(const Real &rhs) const;

	/** Division operator.
	 *
	 *  @param rhs The right hand side of the expression.
	 *
	 *  @return The result of the division. */
	const Real operator/(const Real &rhs) const;

	/** Get the value on the GMP format mpf_t.
	 *
	 *  @return The value as stored on mpf_t type. */
	const mpf_t& getValue() const;
private:
	/** The value. */
	mpf_t value;
};

inline Real::Real(unsigned int precision){
	mpf_init2(value, precision);
}

inline const Real& Real::operator=(double rhs){
	mpf_set_d(value, rhs);

	return *this;
}

inline const Real& Real::operator=(const std::string &rhs){
	mpf_set_str(value, rhs.c_str(), 10);

	return *this;
}

inline const Real Real::operator+(const Real &rhs) const{
	mp_bitcnt_t lhsPrecision = mpf_get_prec(value);
	mp_bitcnt_t rhsPrecision = mpf_get_prec(value);
	Real result((lhsPrecision > rhsPrecision) ? lhsPrecision : rhsPrecision);

	mpf_add(result.value, value, rhs.value);

	return result;
}

inline const Real Real::operator-(const Real &rhs) const{
	mp_bitcnt_t lhsPrecision = mpf_get_prec(value);
	mp_bitcnt_t rhsPrecision = mpf_get_prec(value);
	Real result((lhsPrecision > rhsPrecision) ? lhsPrecision : rhsPrecision);

	mpf_sub(result.value, value, rhs.value);

	return result;
}

inline const Real Real::operator*(const Real &rhs) const{
	mp_bitcnt_t lhsPrecision = mpf_get_prec(value);
	mp_bitcnt_t rhsPrecision = mpf_get_prec(rhs.value);
	Real result((lhsPrecision > rhsPrecision) ? lhsPrecision : rhsPrecision);

	mpf_mul(result.value, value, rhs.value);

	return result;
}

inline const Real Real::operator/(const Real &rhs) const{
	mp_bitcnt_t lhsPrecision = mpf_get_prec(value);
	mp_bitcnt_t rhsPrecision = mpf_get_prec(rhs.value);
	Real result((lhsPrecision > rhsPrecision) ? lhsPrecision : rhsPrecision);

	mpf_div(result.value, value, rhs.value);

	return result;
}

inline const mpf_t& Real::getValue() const{
	return value;
}

}; //End of namespace ArbitraryPrecision
}; //End of namesapce TBTK

#endif
