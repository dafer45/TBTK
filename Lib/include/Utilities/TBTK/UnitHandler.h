/* Copyright 2016 Kristofer Björnson
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
 *  @file UnitHandler.h
 *  @brief Handles conversions between different units.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_HANDLER
#define COM_DAFER45_TBTK_UNIT_HANDLER

#include "TBTK/TBTK.h"
#include "TBTK/Quantity/Constant.h"
#include "TBTK/Quantity/Derived.h"
#include "TBTK/TBTKMacros.h"

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef M_E	//Avoid name clash with math.h macro M_E
	#define M_E_temp M_E
	#undef M_E
#endif

namespace TBTK{

/** @brief Handles conversion between different units.
 *
 *  # Base units and natural units
 *  TBTK does not enforce the use of a specific set of units. Instead it
 *  defines the seven base quantities angle, charge, count, energy, length,
 *  temperature, and time. The UnitHandler allows for seven corresponding base
 *  and natural units to be set for these quantities.
 *
 *  For example, in the following example the base units are set to radians
 *  (rad), Colomb (C), pieces (pcs), milli electron Volt (meV), meter (m),
 *  Kelvin (K), and femtosecond (fs).
 *  ```cpp
 *    UnitHandler::setScales(
 *      {"1 rad", "1.5 C", "5 pcs", "1 meV", "0.3048 m", "1 K", "15 fs"}
 *    );
 *  ```
 *  The natural units are similar to the base units, but also take into account
 *  the scale factor in front of the unit symbols. This means that 3 Colomb
 *  corresponds to three base units, but only two natural units.
 *
 *  In TBTK, all numbers are interpreted in terms of natural units. This means
 *  that TBTK assumes that any number passed to one of its functions is given
 *  in natural units. The default natural units are 1 rad, 1 C, 1 pcs, 1 eV,
 *  1 m, 1 K, and 1 s. If another set of natural units is preferred, the call
 *  to *UnitHandler::setScales()* should occur at the start of the program,
 *  just after TBTK has been initialized.
 *
 *  # Base units
 *  TBTK defines a number of base units for the six @link Quantity::Base Base
 *  Qauntities@endlink. Each can be refered to using its string representation
 *  as above, or using an enum class variable. The later is used in functions
 *  that are potentially performance critical. For exampel, it is possible to
 *  use either Joule (J) or (eV) as base unit for the @link Quantity::Base Base
 *  Quantity@endlink Energy. The string representations for these are "J" and
 *  "eV", while the corresponding enum class representations are
 *  Quantity::Energy::Unit::J and Quantity::Energy::Unit::eV. For a complete
 *  list of base units, see the documentation for Quantity::Base.
 *
 *  # Derived units
 *  In addition to the base units, TBTK also defines derived units for
 *  @link Quantity::Derived Derived Quantities@endlink such as mass and
 *  voltage. The units are called derived since they can be defined in terms of
 *  base units. For example, kg ~ eV m^-2 s^2 and V = eV/e. Some derived units
 *  also have string and enum class representations, such as
 *  Quantity::Mass::Unit::kg and "kg" for mass or Quantity::Voltage::Unit::V
 *  and "V" for voltage. A full list of derived quantities and their
 *  corresponding units, see the documentation for Quantity::Derived.
 *
 *  # Requesting constants
 *  It is possible to request the value of constants in the currently set base
 *  and natural units using
 *  ```cpp
 *    double boltzmannConstantInBaseUnits
 *      = UnitHandler::getConstantInBaseUnits("k_B");
 *    double boltzmannConstantInNaturalUnits
 *      = UnitHandler::getConstantInNaturalUnits("k_B");
 *  ```
 *  The value in base and natural units will differ by a product of scale
 *  factors that accounts for the difference between the two types of units.
 *  The currently available constants are
 *  Constant                | Symbol      | Source*
 *  ------------------------|-------------|-------
 *  Unit charge (positive)  | "e"         | 1
 *  Speed of light          | "c"         | 1
 *  Avogadros number        | "N_A"       | 1
 *  Bohr radius             | "a_0"       | 1
 *  Planck constant         | "h"         | 1
 *  Boltzmann constant      | "k_B"       | 1
 *  Electron mass           | "m_e"       | 2
 *  Proton mass             | "m_p"       | 2
 *  Vacuum permeability     | "mu_0"      | 2
 *  Vacuum permittivity     | "epsilon_0" | 2
 *  Reduced Planck constant | "hbar"      | 3
 *  Bohr magneton           | "mu_B"      | 3
 *  Nuclear magneton        | "mu_N"      | 3
 *
 *  *The sources for the constants are:
 *  -# The International System of Units (SI) 9th Edition. Bureau International
 *  des Poids et Mesures. 2019.
 *  -# The NIST reference on Constants, Units, and Uncertainty.
 *  https://physics.nist.gov/cuu/Constants/index.html.
 *  -# Calculated from the other constants.
 *
 *  # Converting between base and natural units.
 *  Assume that the scales have been set such that the base unit is 1 m and
 *  the natural unit is 1 foot (0.3048 m). It is then possible to convert
 *  between the two units as follows.
 *  ```cpp
 *    double tenMetersInFeet
 *      = UnitHandler::convertBaseToNatural<Quantity::Length>(10);
 *    double tenFeetInMeters
 *      = UnitHandler::convertNaturalToBase<Quantity::Length>(10);
 *  ```
 *  Similar calls can be used to convert between other quantities between base
 *  and natural units by replacing Quantity::Length with the corresponding
 *  Quantity.
 *
 *  # Conversion between arbitrary units and base and natural units
 *  It is also posible to convert between an arbitrary (not currently set) unit
 *  and the (currently set) base and natural units. For example, from "nm" to
 *  "ft", or "V" to C^-1 meV. The functions for this are
 *  ```cpp
 *    double oneVoltInBaseUnits
 *      = UnitHandler::convertArbitraryToBase<Quantity::Voltage>(
 *        1,
 *        Quantity::Voltage::Unit::V
 *      );
 *    double oneVoltInNaturalUnits
 *      = UnitHandler::convertArbitraryToNatural<Quantity::Voltage>(
 *        1,
 *        Quantity::Voltage::Unit::V
 *      );
 *    double oneBaseUnitOfVoltageInVolt
 *      = UnitHandler::convertBaseToVoltage<Quantity::Voltage>(
 *        1,
 *        Quantity::Voltage::Unit::V
 *      );
 *    double oneNaturalVoltageUnitInVolt
 *      = UnitHandler::convertNaturalToArbitrary<Quantity::Voltage>(
 *        1,
 *        Quantity::Voltage::Unit::V
 *      );
 *  ```
 *
 *  # Unit strings
 *  The UnitHandler also allows for a string representation of the base unit
 *  for constants and Quantities to be extracted. Since natural units can come
 *  with an arbitrary numerical factor, unit strings can not be obtained for
 *  natural units. This is on of the reasons why conversion to and from base
 *  units is of interest. All calculations are done in natural units, but
 *  automatic generation of unit strings can only be done for base units. Note,
 *  however, that as long as no scale factor is used, the base and natural
 *  units are the same.
 *
 *  For example, the unit string for Boltzmann constant can be obtained using
 *  ```cpp
 *    std::string unitString = UnitHandler::getUnitString("k_B");
 *  ```
 *  The unit string for voltage can be obtained using
 *  ```cpp
 *    std::string unitString = UnitHandler::getUnitString<Quantity::Voltage>();
 *  ```
 *
 *  # Example
 *  \snippet Utilities/UnitHandler.cpp UnitHandler
 *  ## Output
 *  \snippet output/Utilities/UnitHandler.txt UnitHandler */
class UnitHandler{
public:
	/** Get physical constant in base units. */
	static double getConstantInBaseUnits(const std::string &name);

	/** Get physical constant in natural units. */
	static double getConstantInNaturalUnits(const std::string &name);

	/** Set scales. */
	static void setScales(const std::vector<std::string> &scales);

	/** Convert from natural units to base units. */
	template<typename Quantity>
	static double convertNaturalToBase(double value);

	/** Convert from base units to natural units. */
	template<typename Quantity>
	static double convertBaseToNatural(double value);

	/** @name Convert arbitrary to base
	 *  @{
	 *  Convert from arbitrary units to base units. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		double
	>::type convertArbitraryToBase(
		double value,
		typename Quantity::Unit unit
	);
	template<typename Quantity>
	static typename std::enable_if<
		!Quantity::IsBaseQuantity::value,
		double
	>::type convertArbitraryToBase(
		double value,
		typename Quantity::Unit unit
	);
	/** @} */

	/** @name Convert base to arbitrary
	 *  @{
	 * Convert from base units to arbitrary units. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		double
	>::type convertBaseToArbitrary(
		double value,
		typename Quantity::Unit unit
	);
	template<typename Quantity>
	static typename std::enable_if<
		!Quantity::IsBaseQuantity::value,
		double
	>::type convertBaseToArbitrary(
		double value,
		typename Quantity::Unit unit
	);
	/** @} */

	/** @name Convert arbitrary to natural
	 *  @{
	 * Convert from arbitrary units to natural units. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		double
	>::type convertArbitraryToNatural(
		double value,
		typename Quantity::Unit unit
	);
	template<typename Quantity>
	static typename std::enable_if<
		!Quantity::IsBaseQuantity::value,
		double
	>::type convertArbitraryToNatural(
		double value,
		typename Quantity::Unit unit
	);
	/** @} */

	/** @name Convert natural to arbitrary
	 *  @{
	 * Convert from natural units to arbitrary units. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		double
	>::type convertNaturalToArbitrary(
		double value,
		typename Quantity::Unit unit
	);
	template<typename Quantity>
	static typename std::enable_if<
		!Quantity::IsBaseQuantity::value,
		double
	>::type convertNaturalToArbitrary(
		double value,
		typename Quantity::Unit unit
	);
	/** @} */

	/** @name Get unit string for quantity
	 *  @{
	 *  Get the unit string for the given Quantity in the currently set
	 *  base units.
	 *
	 *  @return string representation of the currently set unit for the
	 *  given Quantity. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		std::string
	>::type getUnitString();
	template<typename Quantity>
	static typename std::enable_if<
		!Quantity::IsBaseQuantity::value,
		std::string
	>::type getUnitString();
	/** @} */

	/** Get the unit string for the given constant.
	 *
	 *  @param constantName The name of the constant.
	 *
	 *  @return The unit string in the current base units. */
	static std::string getUnitString(const std::string &constantName);
private:
	/** Physical constants in the default units K, s, m, eV, C, pcs. */
	static std::map<
		std::string,
		Quantity::Constant
	> constantsDefaultUnits;

	/** Physical constans in the current base units. */
	static std::map<std::string, double> constantsBaseUnits;

	/** Currently set units. */
	static std::tuple<
		Quantity::Angle::Unit,
		Quantity::Charge::Unit,
		Quantity::Count::Unit,
		Quantity::Energy::Unit,
		Quantity::Length::Unit,
		Quantity::Temperature::Unit,
		Quantity::Time::Unit
	> units;

	/** Currently set scales. */
	static std::tuple<
		double,
		double,
		double,
		double,
		double,
		double,
		double
	> scales;

	/** Set unit. */
	template<typename Quantity>
	static void setUnit(typename Quantity::Unit unit);

	/** Function for indexing into the tuple units using compile time
	 *  Quatity names. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		typename Quantity::Unit&
	>::type getUnit();

	/** Set scale. */
	template<typename Quantity>
	static void setScale(double scale);

	/** Function for indexing into the tuple scales using compile time
	 *  Quantity names. */
	template<typename Quantity>
	static double& getScale();

	/** Set scale. */
	template<typename Quantity>
	static void setScale(double scale, typename Quantity::Unit unit);

	/** Set scale. */
	template<typename Quantity>
	static void setScale(const std::string &scale);

	/** Update contants to reflect the current base units. */
	static void updateConstants();

	/** Get the conversion factor needed to go from the default unit to the
	 *  currently set unit. */
	template<typename Quantity>
	static double getConversionFactor();

	/** Get the conversion factor needed to go from the default unit to the
	 *  given unit. */
	template<typename Quantity>
	static double getConversionFactor(typename Quantity::Unit unit);

	/** Converts a string into a corresponding Unit. */
	template<typename Quantity>
	static typename Quantity::Unit getUnit(const std::string &unit);

	/** Get the unit string for the given combination of Base Quantity
	 *  exponents in the currently set base units. */
	static std::string getUnitString(
		int angleExponent,
		int chargeExponent,
		int countExponent,
		int energyExponent,
		int lengthExponent,
		int temperatureExponent,
		int timeExponent
	);

	/** Get the unit string for the given Base Quantity in the currently
	 *  set base units. */
	template<typename Quantity>
	static typename std::enable_if<
		Quantity::IsBaseQuantity::value,
		std::string
	>::type getUnitString(int exponent);

	/** Initialize the UnitHandler. */
	static void initialize();

	/** The Context is a friend of the UnitHandler to allow it to
	 *  initialize the UnitHandler. */
	friend void Initialize();
};

template<typename Quantity>
void UnitHandler::setUnit(typename Quantity::Unit unit){
	double oldConversionFactor = getConversionFactor<Quantity>();
	getUnit<Quantity>() = unit;
	double newConversionFactor = getConversionFactor<Quantity>();
	getScale<Quantity>() *= newConversionFactor/oldConversionFactor;
	updateConstants();
}

template<>
inline Quantity::Angle::Unit& UnitHandler::getUnit<Quantity::Angle>(
){
	return std::get<0>(units);
}

template<>
inline Quantity::Charge::Unit& UnitHandler::getUnit<Quantity::Charge>(
){
	return std::get<1>(units);
}

template<>
inline Quantity::Count::Unit& UnitHandler::getUnit<Quantity::Count>(
){
	return std::get<2>(units);
}

template<>
inline Quantity::Energy::Unit& UnitHandler::getUnit<Quantity::Energy>(
){
	return std::get<3>(units);
}

template<>
inline Quantity::Length::Unit& UnitHandler::getUnit<Quantity::Length>(
){
	return std::get<4>(units);
}

template<>
inline Quantity::Temperature::Unit& UnitHandler::getUnit<
	Quantity::Temperature
>(){
	return std::get<5>(units);
}

template<>
inline Quantity::Time::Unit& UnitHandler::getUnit<Quantity::Time>(
){
	return std::get<6>(units);
}

template <typename Quantity>
void UnitHandler::setScale(double scale, typename Quantity::Unit unit){
	setUnit<Quantity>(unit);
	setScale<Quantity>(scale);
}

inline void UnitHandler::setScales(const std::vector<std::string> &scales){
	TBTKAssert(
		scales.size() == 7,
		"UnitHandler::setScales()",
		"'scales' must contain seven strings.",
		""
	);

	setScale<Quantity::Angle>(scales[0]);
	setScale<Quantity::Charge>(scales[1]);
	setScale<Quantity::Count>(scales[2]);
	setScale<Quantity::Energy>(scales[3]);
	setScale<Quantity::Length>(scales[4]);
	setScale<Quantity::Temperature>(scales[5]);
	setScale<Quantity::Time>(scales[6]);
	updateConstants();
}

template<typename Quantity>
double UnitHandler::convertNaturalToBase(double value){
	return value*getScale<Quantity>();
}

template<typename Quantity>
double UnitHandler::convertBaseToNatural(double value){
	return value/getScale<Quantity>();
}

template<typename Quantity>
typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertArbitraryToBase(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
	)/getConversionFactor<Quantity>(unit);
}

template<typename Quantity>
typename std::enable_if<
	!Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertArbitraryToBase(
	double value,
	typename Quantity::Unit unit
){
	return value*Quantity::getConversionFactor(
		getUnit<TBTK::Quantity::Angle>(),
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
	)/Quantity::getConversionFactor(unit);
}

template<typename Quantity>
typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertBaseToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
		unit
	)/getConversionFactor<Quantity>();
}

template<typename Quantity>
typename std::enable_if<
	!Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertBaseToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	return value*Quantity::getConversionFactor(
		unit
	)/Quantity::getConversionFactor(
		getUnit<TBTK::Quantity::Angle>(),
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
	);
}

template<typename Quantity>
typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertArbitraryToNatural(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
	)/(getConversionFactor<Quantity>(unit)*getScale<Quantity>());
}

template<typename Quantity>
typename std::enable_if<
	!Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertArbitraryToNatural(
	double value,
	typename Quantity::Unit unit
){
	double result = value*Quantity::getConversionFactor(
		getUnit<TBTK::Quantity::Angle>(),
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
	)/Quantity::getConversionFactor(unit);

	result /= pow(
		getScale<TBTK::Quantity::Angle>(),
		Quantity::getExponent(TBTK::Quantity::Angle())
	);
	result /= pow(
		getScale<TBTK::Quantity::Charge>(),
		Quantity::getExponent(TBTK::Quantity::Charge())
	);
	result /= pow(
		getScale<TBTK::Quantity::Count>(),
		Quantity::getExponent(TBTK::Quantity::Count())
	);
	result /= pow(
		getScale<TBTK::Quantity::Energy>(),
		Quantity::getExponent(TBTK::Quantity::Energy())
	);
	result /= pow(
		getScale<TBTK::Quantity::Length>(),
		Quantity::getExponent(TBTK::Quantity::Length())
	);
	result /= pow(
		getScale<TBTK::Quantity::Temperature>(),
		Quantity::getExponent(TBTK::Quantity::Temperature())
	);
	result /= pow(
		getScale<TBTK::Quantity::Time>(),
		Quantity::getExponent(TBTK::Quantity::Time())
	);

	return result;
}

template<typename Quantity>
typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertNaturalToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	return value*getScale<Quantity>()*getConversionFactor<Quantity>(
		unit
	)/getConversionFactor<Quantity>();
}

template<typename Quantity>
typename std::enable_if<
	!Quantity::IsBaseQuantity::value,
	double
>::type UnitHandler::convertNaturalToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	double result = value*Quantity::getConversionFactor(
		unit
	)/Quantity::getConversionFactor(
		getUnit<TBTK::Quantity::Angle>(),
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
	);

	result *= pow(
		getScale<TBTK::Quantity::Angle>(),
		Quantity::getExponent(TBTK::Quantity::Angle())
	);
	result *= pow(
		getScale<TBTK::Quantity::Charge>(),
		Quantity::getExponent(TBTK::Quantity::Charge())
	);
	result *= pow(
		getScale<TBTK::Quantity::Count>(),
		Quantity::getExponent(TBTK::Quantity::Count())
	);
	result *= pow(
		getScale<TBTK::Quantity::Energy>(),
		Quantity::getExponent(TBTK::Quantity::Energy())
	);
	result *= pow(
		getScale<TBTK::Quantity::Length>(),
		Quantity::getExponent(TBTK::Quantity::Length())
	);
	result *= pow(
		getScale<TBTK::Quantity::Temperature>(),
		Quantity::getExponent(TBTK::Quantity::Temperature())
	);
	result *= pow(
		getScale<TBTK::Quantity::Time>(),
		Quantity::getExponent(TBTK::Quantity::Time())
	);

	return result;
}

template<typename Quantity>
double UnitHandler::getConversionFactor(){

	return Quantity::getConversionFactor(getUnit<Quantity>());
}

template<typename Quantity>
double UnitHandler::getConversionFactor(typename Quantity::Unit unit){
	return Quantity::getConversionFactor(unit);
}

template<typename Quantity>
inline void UnitHandler::setScale(double scale){
	getScale<Quantity>() = scale;
}

template<typename Quantity>
inline void UnitHandler::setScale(const std::string &scale){
	std::stringstream stream(scale);
	std::vector<std::string> components;
	std::string word;
	while(std::getline(stream, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setScale()",
		"Invalid scale string '" << scale << "'.",
		"The string must be on the format '[scale] [unit]', e.g. '1 K'"
	);

	double value;
	try{
		value = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setScale()",
			"Unable to parse '" << components[0] << "' as a"
			<< " double.",
			"The string has to be on the format '[scale] [unit]',"
			<< " e.g. '1 K'."
		);
	}

	typename Quantity::Unit unit = getUnit<Quantity>(components[1]);

	setScale<Quantity>(value, unit);
}

template<>
inline double& UnitHandler::getScale<Quantity::Angle>(){
	return std::get<0>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Charge>(){
	return std::get<1>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Count>(){
	return std::get<2>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Energy>(){
	return std::get<3>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Length>(){
	return std::get<4>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Temperature>(){
	return std::get<5>(scales);
}

template<>
inline double& UnitHandler::getScale<Quantity::Time>(){
	return std::get<6>(scales);
}

template<typename Quantity>
inline typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	std::string
>::type UnitHandler::getUnitString(){
	return Quantity::getUnitString(getUnit<Quantity>());
}

template<typename Quantity>
inline typename std::enable_if<
	!Quantity::IsBaseQuantity::value,
	std::string
>::type UnitHandler::getUnitString(){
	return getUnitString(
		static_cast<int>(Quantity::Exponent::Angle),
		static_cast<int>(Quantity::Exponent::Charge),
		static_cast<int>(Quantity::Exponent::Count),
		static_cast<int>(Quantity::Exponent::Energy),
		static_cast<int>(Quantity::Exponent::Length),
		static_cast<int>(Quantity::Exponent::Temperature),
		static_cast<int>(Quantity::Exponent::Time)
	);
}

template<typename Quantity>
typename Quantity::Unit UnitHandler::getUnit(const std::string &unit){
	return Quantity::getUnit(unit);
}

inline std::string UnitHandler::getUnitString(
	int angleExponent,
	int chargeExponent,
	int countExponent,
	int energyExponent,
	int lengthExponent,
	int temperatureExponent,
	int timeExponent
){
	std::string result;
	result += getUnitString<Quantity::Angle>(angleExponent);
	result += getUnitString<Quantity::Charge>(chargeExponent);
	result += getUnitString<Quantity::Count>(countExponent);
	result += getUnitString<Quantity::Energy>(energyExponent);
	result += getUnitString<Quantity::Length>(lengthExponent);
	result += getUnitString<Quantity::Temperature>(temperatureExponent);
	result += getUnitString<Quantity::Time>(timeExponent);

	if(result.size() != 0)
		result.pop_back();

	return result;
}

template<typename Quantity>
inline typename std::enable_if<
	Quantity::IsBaseQuantity::value,
	std::string
>::type UnitHandler::getUnitString(int exponent){
	std::string result;
	if(exponent != 0){
		result += getUnitString<Quantity>();
		if(exponent != 1)
			result += "^" + std::to_string(exponent);
		result += " ";
	}
	return result;
}

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
