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
 *  The UnitHandler handles conversion between the 'natural units' used in the
 *  calculations and standard 'base units'. The base quantities are
 *  Temperature, time, length, energy, charge, and count, and the default
 *  units are Kelvin (K), second (s), meter (m), electron Volt (eV),
 *  Coulomb (C), and pieces (pcs).
 *
 *  The current base units can be changed with setXXXUnit(), where \f$ XXX \in
 *  \{Temperature, Time, Length, Energy, Charge, Count\}\f$, and subsequent
 *  calls will take input paramaters and return output parameters in
 *  corresponding units.
 *
 *  The functions setXXXScale() can be used to specify the scale of the 'ruler'
 *  with which parameters in the calculation are meassured. This sets the
 *  scale which relates the natrual units to the base units. For example:
 *
 *  UnitHandler::setEnergyUnits(UnitHandler::EnergyUnit:meV);
 *  UnitHandler::setEnergyScale(1.47);
 *
 *  Here the energy base unit is first set to meV, and the scale subsequently
 *  set to 1.47meV. The natural unit 1.0, now corresponds to the base unit
 *  1.47meV. An energy parameter with value 1.0 will therefore be
 *  interpreted as having the physical value 1.47meV whenever a multiplication
 *  with a physical constant is required to generate a unitless number.
 *
 *  Note that the order of the above calls are important. Setting the energy
 *  scale to 1.47 while the default energy unit eV is used and then changing to
 *  meV as units will result in the scale becoming 1.47eV = 1470meV.
 *
 *  In addition to base units and natural units, there are also derived units.
 *  These are used to convert to and from units not included among the base
 *  units, such as kg for mass. That a u unit is derived means that it can be
 *  expressed in terms of the already existing base units. For example: kg =
 *  eVs^2/m^2. */
class UnitHandler{
public:
	/** Get physical constant in base units. */
	static double getConstantBaseUnits(const std::string &name);

	/** Get physical constant in natural units. */
	static double getConstantNaturalUnits(const std::string &name);

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

	/** Get the unit string for the given Quantity in the currently set
	 *  base units.
	 *
	 *  @return string representation of the currently set unit for the
	 *  given Quantity. */
	template<typename Quantity>
	static std::string getUnitString();

	/** Get mass unit string.
	 *
	 *  @return string representation of the derived mass unit in terms of
	 *  the currently set base units. */
	static std::string getMassUnitString();

	/** Get mass unit string.
	 *
	 *  @return string representation of the derived magnetic field unit in
	 *  terms of the currently set base units. */
	static std::string getMagneticFieldUnitString();

	/** Get voltage unit string.
	 *
	 *  @return string representation of the derived voltage unit in terms
	 *  of the currently set base units. */
	static std::string getVoltageUnitString();

	/** Get Planck constant unit string
	 *
	 *  @return string representation of the unit for the Planck constant.
	 */
	static std::string getHBARUnitString();

	/** Get Boltzmann constant unit string.
	 *
	 *  @return string representation of the unit for the Boltzmann
	 *  constant. */
	static std::string getK_BUnitString();

	/** Get elementary charge unit string.
	 *
	 *  @return string representation of the unit for the elementary
	 *  charge.*/
	static std::string getEUnitString();

	/** Get speed of light unit string.
	 *
	 *  @return string representation of the unit for the speed of light.
	 */
	static std::string getCUnitString();

	/** GetAvogadros number unit string.
	 *
	 *  @return string representation of the unit for Avogadros number. */
	static std::string getN_AUnitString();

	/** Get electron mass unit string.
	 *
	 *  @return string representation of the unit for the electron mass. */
	static std::string getM_eUnitString();

	/** Get proton mass unit string.
	 *
	 *  @return string representation of the unit for the proton mass. */
	static std::string getM_pUnitString();

	/** Get Bohr magneton unit string.
	 *
	 *  @return string representation of the unit for the Bohr magneton. */
	static std::string getMu_BUnitString();

	/** Get nuclear magneton unit string.
	 *
	 *  @return string representation of the unit for the nuclear magneton. */
	static std::string getMu_nUnitString();

	/** Get vacuum permeability unit string.
	 *
	 *  @return string representation of the unit for the vacuum
	 *  permeability. */
	static std::string getMu_0UnitString();

	/** Get vacuum permittivity unit string
	 *
	 *  @return string representation of the unit for the vacuum
	 *  permittivity. */
	static std::string getEpsilon_0UnitString();

	/** Get the Bohr radius unit string
	 *
	 *  @return string representation of the unit for the Bohr radius. */
	static std::string getA_0UnitString();
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
		double
	> scales;

	/** Set unit. */
	template<typename Quantity>
	static void setUnit(typename Quantity::Unit unit);

	/** Function for indexing into the tuple units using compile time
	 *  Quatity names. */
	template<typename Quantity>
	constexpr static typename Quantity::Unit& getUnit();

	/** Set scale. */
	template<typename Quantity>
	static void setScale(double scale);

	/** Function for indexing into the tuple scales using compile time
	 *  Quantity names. */
	template<typename Quantity>
	constexpr static double& getScale();

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
inline constexpr Quantity::Charge::Unit& UnitHandler::getUnit<Quantity::Charge>(
){
	return std::get<0>(units);
}

template<>
inline constexpr Quantity::Count::Unit& UnitHandler::getUnit<Quantity::Count>(
){
	return std::get<1>(units);
}

template<>
inline constexpr Quantity::Energy::Unit& UnitHandler::getUnit<Quantity::Energy>(
){
	return std::get<2>(units);
}

template<>
inline constexpr Quantity::Length::Unit& UnitHandler::getUnit<Quantity::Length>(
){
	return std::get<3>(units);
}

template<>
inline constexpr Quantity::Temperature::Unit& UnitHandler::getUnit<
	Quantity::Temperature
>(){
	return std::get<4>(units);
}

template<>
inline constexpr Quantity::Time::Unit& UnitHandler::getUnit<Quantity::Time>(
){
	return std::get<5>(units);
}

template <typename Quantity>
void UnitHandler::setScale(double scale, typename Quantity::Unit unit){
	setUnit<Quantity>(unit);
	setScale<Quantity>(scale);
}

inline void UnitHandler::setScales(const std::vector<std::string> &scales){
	TBTKAssert(
		scales.size() == 6,
		"UnitHandler::setScales()",
		"'scales' must contain six strings.",
		""
	);

	setScale<Quantity::Charge>(scales[0]);
	setScale<Quantity::Count>(scales[1]);
	setScale<Quantity::Energy>(scales[2]);
	setScale<Quantity::Length>(scales[3]);
	setScale<Quantity::Temperature>(scales[4]);
	setScale<Quantity::Time>(scales[5]);
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
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
	)/Quantity::getConversionFactor(unit);

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
		getUnit<TBTK::Quantity::Charge>(),
		getUnit<TBTK::Quantity::Count>(),
		getUnit<TBTK::Quantity::Energy>(),
		getUnit<TBTK::Quantity::Length>(),
		getUnit<TBTK::Quantity::Temperature>(),
		getUnit<TBTK::Quantity::Time>()
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
inline constexpr double& UnitHandler::getScale<Quantity::Charge>(){
	return std::get<0>(scales);
}

template<>
inline constexpr double& UnitHandler::getScale<Quantity::Count>(){
	return std::get<1>(scales);
}

template<>
inline constexpr double& UnitHandler::getScale<Quantity::Energy>(){
	return std::get<2>(scales);
}

template<>
inline constexpr double& UnitHandler::getScale<Quantity::Length>(){
	return std::get<3>(scales);
}

template<>
inline constexpr double& UnitHandler::getScale<Quantity::Temperature>(){
	return std::get<4>(scales);
}

template<>
inline constexpr double& UnitHandler::getScale<Quantity::Time>(){
	return std::get<5>(scales);
}

template<typename Quantity>
inline std::string UnitHandler::getUnitString(){
	return Quantity::getUnitString(getUnit<Quantity>());
}

template<typename Quantity>
typename Quantity::Unit UnitHandler::getUnit(const std::string &unit){
	return Quantity::getUnit(unit);
}

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
