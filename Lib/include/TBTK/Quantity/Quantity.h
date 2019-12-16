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
 *  @file Quantity.h
 *  @brief Quantity.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_QUANTITY_QUANTITY
#define COM_DAFER45_TBTK_QUANTITY_QUANTITY

#include "TBTK/Real.h"
#include "TBTK/TBTKMacros.h"

#include <map>
#include <string>

namespace TBTK{
namespace Quantity{

void initializeBaseQuantities();

/** @brief Quantity.
 *
 *  Base class for Quantities.
 */
template<typename Units, typename Exponents>
class Quantity : public Real{
public:
	using Unit = Units;
	using Exponent = Exponents;

	/** Default constructor. */
	Quantity(){};

	/** Constructs a Quantity from a double. */
	Quantity(double value) : Real(value){};

	/** Get unit string for the given Unit. */
	static std::string getUnitString(Unit unit);

	/** Convert a string to a Unit. */
	static Unit getUnit(const std::string &unit);

	/** Get the conversion factor for converting from the reference unit to
	 *  the given unit. */
	static double getConversionFactor(Unit unit);
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

	friend void initializeBaseQuantities();
};

template<typename Units, typename Exponents>
std::string Quantity<Units, Exponents>::getUnitString(Unit unit){
	try{
		return conversionTable.unitToString.at(unit);
	}
	catch(const std::out_of_range &e){
		TBTKExit(
			"Quantity::getUnitString()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			""
		);
		return "";
	}
}

template<typename Units, typename Exponents>
typename Quantity<Units, Exponents>::Unit Quantity<Units, Exponents>::getUnit(
	const std::string &unit
){
	try{
		return conversionTable.stringToUnit.at(unit);
	}
	catch(const std::out_of_range &e){
		TBTKExit(
			"Quantity::getUnit()",
			"Unknown unit '" << unit << "'.",
			""
		);
	}
}

template<typename Units, typename Exponents>
double Quantity<Units, Exponents>::getConversionFactor(Unit unit){
	try{
		Streams::out << "1\t" << &conversionTable << "\n";
		for(auto entry : conversionTable.conversionFactors)
			Streams::out << static_cast<int>(entry.first) << "\t" << entry.second << "\n";
		Streams::out << "2\n";

		return conversionTable.conversionFactors.at(unit);
	}
	catch(const std::out_of_range &e){
		TBTKExit(
			"Quantity::getConversionFactor()",
			"Unknown unit '" << static_cast<int>(unit) << "'.",
			""
		);
	}
}

}; //End of namesapce Quantity
}; //End of namesapce TBTK

#endif

