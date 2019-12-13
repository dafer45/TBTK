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
 *  @file Length.h
 *  @brief Length.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_LENGTH
#define COM_DAFER45_TBTK_QUANTITY_LENGTH

#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

/** @brief Length.
 *
 *  A Length is a Real value which implicitly is assumed to have units of
 *  length. */
class Length : public Real{
public:
	/** Default constructor. */
	Length(){};

	/** Constructs a Quantity from a double. */
	Length(double value) : Real(value){};

	/** Length units (base unit):
	 *  - m - meter
	 *  - mm - millimeter
	 *  - um - micrometer
	 *  - nm - nanometer
	 *  - pm - picometer
	 *  - fm - femtometer
	 *  - am - attometer
	 *  - Ao - Angstrom */
	enum class Unit {m, mm, um, nm, pm, fm, am, Ao};

	/** Get unit string for the given unit. */
	static std::string getUnitString(Unit unit);

	/** Convert a string to a Unit. */
	static Unit getUnit(const std::string &str);

	/** Get the conversion factor for converting from the reference unit to
	 *  the given unit. */
	static double getConversionFactor(Unit unit);
private:
	static std::map<Unit, std::string> unitToString;
	static std::map<std::string, Unit> stringToUnit;
};

}; //End of namesapce Length
}; //End of namesapce TBTK

#endif
