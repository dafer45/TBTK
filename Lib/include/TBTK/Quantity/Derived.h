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
 *  @file Derived.h
 *  @brief Derived.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_DERIVED
#define COM_DAFER45_TBTK_QUANTITY_DERIVED

#include "TBTK/Quantity/Charge.h"
#include "TBTK/Quantity/Count.h"
#include "TBTK/Quantity/Energy.h"
#include "TBTK/Quantity/Length.h"
#include "TBTK/Quantity/Temperature.h"
#include "TBTK/Quantity/Time.h"
#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

/** @brief Derived.
 *
 *  Derived provides the means for defining derived Quantities.
 */
template<typename Units, typename Exponents>
class Derived : public Real{
public:
	using Unit = Units;
	using Exponent = Exponents;

	/** Default constructor. */
	Derived(){};

	/** Constructs a Quantity from a double. */
	Derived(double value) : Real(value){};

	/** Get unit string for the given Unit. */
	static std::string getUnitString(Unit unit);

	/** Convert a string to a Unit. */
	static Unit getUnit(const std::string &unit);

	/** Get the conversion factor for converting from the reference unit to
	 *  the given unit. */
	static double getConversionFactor(Unit unit);

	/** Get the conversion factor for converting from the reference unit to
	 *  the given units. */
	static double getConversionFactor(
		Quantity::Charge::Unit chargeUnit,
		Quantity::Count::Unit countUnit,
		Quantity::Energy::Unit energyUnit,
		Quantity::Length::Unit lengthUnit,
		Quantity::Temperature::Unit temperatureUnit,
		Quantity::Time::Unit timeUnit
	);
private:
	static class ConversionTable{
	public:
		std::map<Unit, std::string> unitToString;
		std::map<std::string, Unit> stringToUnit;
		std::map<Unit, double> conversionFactors;

		ConversionTable(
			const std::map<
				Unit,
				std::pair<std::string, double>
			> &conversionTable
		){
			for(auto entry : conversionTable){
				unitToString[entry.first] = entry.second.first;
				stringToUnit[entry.second.first] = entry.first;
				conversionFactors[entry.first]
					= entry.second.second;
			}
		}
	} conversionTable;
};

template<typename Units, typename Exponents>
std::string Derived<Units, Exponents>::getUnitString(Unit unit){
	return conversionTable.unitToString.at(unit);
}

template<typename Units, typename Exponents>
typename Derived<Units, Exponents>::Unit Derived<Units, Exponents>::getUnit(
	const std::string &unit
){
	try{
		return conversionTable.stringToUnit.at(unit);
	}
	catch(const std::out_of_range &e){
		TBTKExit(
			"Derived::getUnit()",
			"Unknown unit '" << unit << "'.",
			""
		);
	}
}

template<typename Units, typename Exponents>
double Derived<Units, Exponents>::getConversionFactor(Unit unit){
	return conversionTable.conversionFactors.at(unit);
}

template<typename Units, typename Exponents>
double Derived<Units, Exponents>::getConversionFactor(
	Charge::Unit chargeUnit,
	Count::Unit countUnit,
	Energy::Unit energyUnit,
	Length::Unit lengthUnit,
	Temperature::Unit temperatureUnit,
	Time::Unit timeUnit
){
	return pow(
		Charge::getConversionFactor(chargeUnit),
		Exponent::Charge
	)*pow(
		Count::getConversionFactor(countUnit),
		Exponent::Count
	)*pow(
		Energy::getConversionFactor(energyUnit),
		Exponent::Energy
	)*pow(
		Length::getConversionFactor(lengthUnit),
		Exponent::Length
	)*pow(
		Temperature::getConversionFactor(temperatureUnit),
		Exponent::Temperature
	)*pow(
		Time::getConversionFactor(timeUnit),
		Exponent::Time
	);
}

enum class MassUnit {kg, g, mg, ug, ng, pg, fg, ag, u};
enum class MassExponent {
	Charge = 0,
	Count = 0,
	Energy = 1,
	Length = -2,
	Temperature = 0,
	Time = 2
};
typedef Derived<MassUnit, MassExponent> Mass;

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

