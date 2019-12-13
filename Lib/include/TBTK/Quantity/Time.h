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
 *  @file Time.h
 *  @brief Time.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_TIME
#define COM_DAFER45_TBTK_QUANTITY_TIME

#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

/** @brief Time.
 *
 *  A Time is a Real value which implicitly is assumed to have units of time.
 */
class Time : public Real{
public:
	/** Default constructor. */
	Time(){};

	/** Constructs a Quantity from a double. */
	Time(double value) : Real(value){};

	/** Time units (base unit):
	 *  - s - second
	 *  - ms - millisecond
	 *  - us - microsecond
	 *  - ns - nanosecond
	 *  - ps - picosecond
	 *  - fs - femtosecond
	 *  - as - attosecond */
	enum class Unit {s, ms, us, ns, ps, fs, as};

	/** Get unit string for the given Unit. */
	static std::string getUnitString(Unit unit);

	/** Convert a string to a Unit. */
	static Unit getUnit(const std::string &str);
private:
	static std::map<Unit, std::string> unitToString;
	static std::map<std::string, Unit> stringToUnit;
};

}; //End of namesapce Time
}; //End of namesapce TBTK

#endif
