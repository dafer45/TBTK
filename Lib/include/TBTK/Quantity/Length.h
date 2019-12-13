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

#include "TBTK/json.hpp"

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

	/** Get unit string. */
	static std::string getUnitString(Unit unit);
};

inline std::string Length::getUnitString(Unit unit){
	switch(unit){
		case Quantity::Length::Unit::m:
			return "m";
		case Quantity::Length::Unit::mm:
			return "mm";
		case Quantity::Length::Unit::um:
			return "um";
		case Quantity::Length::Unit::nm:
			return "nm";
		case Quantity::Length::Unit::pm:
			return "pm";
		case Quantity::Length::Unit::fm:
			return "fm";
		case Quantity::Length::Unit::am:
			return "am";
		case Quantity::Length::Unit::Ao:
			return "Ao";
		default:
			return "Unknown unit";
	};
}

}; //End of namesapce Length
}; //End of namesapce TBTK

#endif