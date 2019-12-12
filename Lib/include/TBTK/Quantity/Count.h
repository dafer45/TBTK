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
 *  @file Count.h
 *  @brief Count.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_COUNT
#define COM_DAFER45_TBTK_QUANTITY_COUNT

#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{
namespace Quantity{

/** @brief Count.
 *
 *  A Count is a Real value which implicitly is assumed to have units of count.
 */
class Count : public Real{
public:
	/** Default constructor. */
	Count(){};

	/** Constructs a Quantity from a double. */
	Count(double value) : Real(value){};

	/** Count unit (base unit):
	 * - pcs - pieces
	 * - mol - Mole */
	enum class Unit{pcs, mol};

	/** Get unit string. */
	static std::string getUnitString(Unit unit);
};

inline std::string Count::getUnitString(Unit unit){
	switch(unit){
		case Quantity::Count::Unit::pcs:
			return "pcs";
		case Quantity::Count::Unit::mol:
			return "mol";
		default:
			return "Unkown unit";
	}
}

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

