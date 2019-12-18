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
 *  @brief Base class for @link Quantity Quantities@endlink.
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

/** @brief Base class for @link Quantity Quantitits@endlink.
 *
 *  A Quantity is a Real number that carries compile time information about
 *  units. While it is possible to create Quantity object, their main purpose
 *  is to act as template parameter for the UnitHandler functions.
 *
 *  # Defining new Quantities
 *  *Note: The Quantity template is not meant to be instatiated directly.
 *  Instead the Base and Derived classes that inherit from Quantity are meant
 *  to be instatiated. However, the syntax for instantiating such Quantities is
 *  the same as instantiating a Quantity directly.*
 *
 *  ## Units and exponents
 *  To instantiate a Quantity, we first need to define a set of units for the
 *  Quantity. This is done with the help of an enum class.
 *  ```cpp
 *    enum class VoltageUnit {V, mV};
 *  ```
 *  We also need to define the exponents for the base units Charge, Count,
 *  Energy, Length, Temperature, and Time. In the case of voltage, this is V =
 *  J/C, or Energy/Charge.
 *  ```cpp
 *    enum class Voltage{
 *      Charge = -1,
 *      Count = 0,
 *      Energy = 1,
 *      Length = 0,
 *      Temperature = 0,
 *      Time = 0
 *    };
 *  ```
 *  Having defined these two enum classes, we can define Voltage using
 *  ```cpp
 *    typedef Quantity<VoltageUnit, VoltageExponent> Voltage;
 *  ```
 *  <b>Note that when instantiating Base and Derived units, Quantity in the
 *  line above is replaced by Base and Derived, respectively.</b>
 *
 *  With these definitions, the following symbols that are relevant for
 *  UnitHandler calls are defined:
 *  - Quantity::Voltage
 *  - Quantity::Voltage::Unit::V
 *  - Quantity::Voltage::Unit::mV
 *
 *  ## Creating and initializing a conversion table
 *  We also need to define a table that can be used to convert between string
 *  and enum class representations of the units. The conversion table should
 *  also contain a conversion factor that specifies the amount in the given
 *  unit that corresponds to a single unit of the Quantity in the base default
 *  base units. The default base units are C, pcs, eV, m, K, and s.
 *
 *  The unit for voltage is V = J/C = 1.602176634e-19. We can therefore define
 *  the conversion table as follows.
 *  ```cpp
 *    Quantity<
 *      VoltageUnit,
 *      VoltageExponent
 *    >::ConversionTable Quantity<
 *      VoltageUnit,
 *      VoltageExponent
 *    >::conversionTable({
 *      {Quantity::Voltage::Unit::V, {"V", 1.602176634e-19}},
 *      {Quantity::Voltage::Unit::mV, {"mV", 1.602176634e-19*1e3}},
 *    });
 *  ```
 *    <b>Notes:
 *      - This definition has to be made in a .cpp-file rather than a .h-file
 *      to avoid multiple definitions.
 *      - Even though actual Quantities should by instantiated from the Base
 *      and Derived classes, this definition should always contain the keyword
 *      Quantity. Note the difference with how the typedef is done, where
 *      Quantity is to be replaced by Base or Derived.
 *      - The initialization performed above is only possible if the numerical
 *      constants are hard coded and can be used when defining custom Derived
 *      units. However, the Base and Derived Quantities defined in TBTK uses
 *      constants from Quantity::Constants rather than specifying numerical
 *      values directly. These values are not available until Constants have
 *      been initialized. The conversion table definition and initialization is
 *      therefore separated into two parts as shown below.
 *
 *    </b>
 *
 *  The conversionTable is defined as follows. (Note the empty curly brace,
 *  these are necessary to ensure that the conversion table is created.)
 *  ```cpp
 *    Quantity<
 *      VoltageUnit,
 *      VoltageExponent
 *    >::ConversionTable Quantity<
 *      VoltageUnit,
 *      VoltageExponent
 *    >::conversionTable({});
 *  ```
 *  The following code is then put into the function
 *  *initializeDerivedQuantities()* to initialize the conversion table.
 *  ```cpp
 *    Voltage::conversionTable = Voltage::ConversionTable({
 *      {Voltage::Unit::V, {"V", 1.602176634e-19}},
 *      {Voltage::Unit::mV, {"mV", 1.602176634e-19*1e3}},
 *    });
 *  ```
 *  The conversion tables for Base units are similarly initialized in
 *  *initializeBaseQuantities()*.
 *
 *  It is not possible to define custom Derived units and initialize them from
 *  another function that *initializeDerivedQuantities()*. Custom defined
 *  Derived units therefore either have to be defined using hard coded
 *  constants or be defined by extending the Derived.h and Derived.cpp files
 *  directly.
 *
 *  # Conversion
 *  ## Unit string to unit
 *
 *  ```cpp
 *    Quantity::Voltage::Unit unit = Quantity::Voltage::getUnit("V");
 *  ```
 *  ## Unit to unit string
 *
 *  ```cpp
 *    std::string unitString
 *      = Quantity::Voltage::getUnitString(Quantity::Voltage::Unit::V);
 *  ```
 *  ## Get conversion factor from the default base units to the given unit
 *  ```cpp
 *    double conversionFactor
 *      = Quantity::Voltage::getConversionFactor(Quantity::Voltage::Unit::V);
 *  ```
 *  */
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
protected:
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

