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
 *  @file Constant.h
 *  @brief Numerical constant.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_CONSTANT
#define COM_DAFER45_TBTK_QUANTITY_CONSTANT

#include "TBTK/Quantity/Base.h"
#include "TBTK/Quantity/Derived.h"
#include "TBTK/TBTK.h"

#include <cmath>
#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

class Constant{
public:
	Constant(){};

	template<typename Quantity>
	explicit Constant(double value, Quantity);

	template<typename Quantity>
	int getExponent() const;

	operator double(){
		return value;
	};
private:
	double value;
	std::vector<int> exponents;
};

template<typename Quantity>
Constant::Constant(
	double value,
	Quantity
) :
	value(value),
	exponents(6)
{
	exponents[0] = static_cast<int>(Quantity::Exponent::Charge);
	exponents[1] = static_cast<int>(Quantity::Exponent::Count);
	exponents[2] = static_cast<int>(Quantity::Exponent::Energy);
	exponents[3] = static_cast<int>(Quantity::Exponent::Length);
	exponents[4] = static_cast<int>(Quantity::Exponent::Temperature);
	exponents[5] = static_cast<int>(Quantity::Exponent::Time);
}

template<>
inline int Constant::getExponent<Charge>() const{
	return exponents[0];
}

template<>
inline int Constant::getExponent<Count>() const{
	return exponents[1];
}

template<>
inline int Constant::getExponent<Energy>() const{
	return exponents[2];
}

template<>
inline int Constant::getExponent<Length>() const{
	return exponents[3];
}

template<>
inline int Constant::getExponent<Temperature>() const{
	return exponents[4];
}

template<>
inline int Constant::getExponent<Time>() const{
	return exponents[5];
}

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif
/// @endcond
