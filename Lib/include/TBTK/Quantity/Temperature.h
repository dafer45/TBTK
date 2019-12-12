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
 *  @file Temperature.h
 *  @brief Temperature.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_TEMPERATURE
#define COM_DAFER45_TBTK_QUANTITY_TEMPERATURE

#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{
namespace Quantity{

/** @brief Temperature.
 *
 *  A Temperature is a Real value which implicitly is assumed to have units of
 *  temperature. */
class Temperature : public Real{
public:
	/** Default constructor. */
	Temperature(){};

	/** Constructs a Quantity from a double. */
	Temperature(double value) : Real(value){};

	/** Temperature units (base unit):
	 *  - kK - kilokelvin
	 *  - K - Kelvin
	 *  - mK - millikelvin
	 *  - uK - microkelvin
	 *  - nK - nanokelvin */
	enum class Unit {kK, K, mK, uK, nK};

	/** Get unit string. */
	static std::string getUnitString(Unit unit);
};

inline std::string Temperature::getUnitString(Unit unit){
	switch(unit){
		case Quantity::Temperature::Unit::kK:
			return "kK";
		case Quantity::Temperature::Unit::K:
			return "K";
		case Quantity::Temperature::Unit::mK:
			return "mK";
		case Quantity::Temperature::Unit::uK:
			return "uK";
		case Quantity::Temperature::Unit::nK:
			return "nK";
		default:
			return "Unknown unit";
	};
}

}; //End of namesapce Temperature
}; //End of namesapce TBTK

#endif
