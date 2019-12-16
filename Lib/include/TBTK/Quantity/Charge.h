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
 *  @file Charge.h
 *  @brief Charge.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_CHARGE
#define COM_DAFER45_TBTK_QUANTITY_CHARGE

#include "TBTK/Quantity/Quantity.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

/*enum class ChargeUnit{kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e};
enum class ChargeExponent{
	Charge = 1,
	Count = 0,
	Energy = 0,
	Length = 0,
	Temperature = 0,
	Time = 0
};
typedef Quantity<ChargeUnit, ChargeExponent> Charge;*/

/** @brief Charge.
 *
 *  A Charge is a Real value which implicitly is assumed to have units of
 *  charge. */
//class Charge : public Real{
//public:
	/** Default constructor. */
//	Charge(){};

	/** Constructs a Quantity from a double. */
//	Charge(double value) : Real(value){};

	/** Charge units (base unit):
	 * - kC - kilocoulomb
	 * - C - Coulomb
	 * - mC - millicoulomb
	 * - uC - microcoulomb
	 * - nC - nanocoulomb
	 * - pC - picocoulomb
	 * - fC - femtocoulomb
	 * - aC - attocoulomb
	 * - Te - terrae
	 * - Ge - gigae
	 * - Me - megae
	 * - ke - kiloe
	 * - e - e (elementary charge) */
//	enum class Unit{kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e};

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

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif
