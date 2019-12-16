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

#include "TBTK/Quantity/Quantity.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

enum class TemperatureUnit{kK, K, mK, uK, nK};
enum class TemperatureExponent{
	Charge = 0,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 1,
	Time = 0
};
typedef Quantity<TemperatureUnit, TemperatureExponent> Temperature;

/** @brief Temperature.
 *
 *  A Temperature is a Real value which implicitly is assumed to have units of
 *  temperature. */
//class Temperature : public Real{
//public:
	/** Default constructor. */
//	Temperature(){};

	/** Constructs a Quantity from a double. */
//	Temperature(double value) : Real(value){};

	/** Temperature units (base unit):
	 *  - kK - kilokelvin
	 *  - K - Kelvin
	 *  - mK - millikelvin
	 *  - uK - microkelvin
	 *  - nK - nanokelvin */
//	enum class Unit {kK, K, mK, uK, nK};

	/** Get unit string for the given Unit. */
//	static std::string getUnitString(Unit unit);

	/** Convert a string to a Unit. */
//	static Unit getUnit(const std::string &str);

	/** Get the conversion factor for converting from the reference unit to
	 *  the given unit. */
//	static double getConversionFactor(Unit unit);
//private:
//	static std::map<Unit, std::string> unitToString;
//	static std::map<std::string, Unit> stringToUnit;
//};

}; //End of namesapce Temperature
}; //End of namesapce TBTK

#endif
