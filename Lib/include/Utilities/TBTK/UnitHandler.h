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

#include "TBTK/Quantity/Charge.h"
#include "TBTK/Quantity/Count.h"
#include "TBTK/Quantity/Energy.h"
#include "TBTK/Quantity/Length.h"
#include "TBTK/Quantity/Temperature.h"
#include "TBTK/Quantity/Time.h"
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
	/** Mass units (derived unit):<br/>
	 *	kg - kilogram<br/>
	 *	g - gram<br/>
	 *	mg - milligram<br/>
	 *	ug - microgram<br/>
	 *	ng - nanogram<br/>
	 *	pg - picogram<br/>
	 *	fg - femtogram<br/>
	 *	ag - attogram<br/>
	 *	u - atomic mass */
	enum class MassUnit{kg, g, mg, ug, ng, pg, fg, ag, u};

	/** Magnetic unit (derived unit):<br/>
	 *	MT - megatesla<br/>
	 *	kT - kilotesla<br/>
	 *	T - Tesla<br/>
	 *	mT - millitesla<br/>
	 *	uT - microtesla<br/>
	 *	nT - nanotesla<br/>
	 *	GG - gigagauss<br/>
	 *	MG - megagauss<br/>
	 *	kG - kilogauss<br/>
	 *	G - Gauss<br/>
	 *	mG - milligauss<br/>
	 *	uG - microgauss */
	enum class MagneticFieldUnit{
		MT, kT, T, mT, uT, nT, GG, MG, kG, G, mG, uG
	};

	/** Voltage unit (derived unit):<br/>
	 *	GV - kilovolt<br/>
	 *	MV - kilovolt<br/>
	 *	kV - kilovolt<br/>
	 *	V - volt<br/>
	 *	mV - millivolt</br>
	 *	uV - millivolt<br/>
	 *	nV - millivolt */
	enum class VoltageUnit{
		GV, MV, kV, V, mV, uV, nV
	};

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

	/** Convert arbitrary units to base units. */
	template<typename Quantity>
	static double convertArbitraryToBase(
		double value,
		typename Quantity::Unit unit
	);

	/** Convert base units to arbitrary units. */
	template<typename Quantity>
	static double convertBaseToArbitrary(
		double value,
		typename Quantity::Unit unit
	);

	/** Convert arbitrary units to natural units. */
	template<typename Quantity>
	static double convertArbitraryToNatural(
		double value,
		typename Quantity::Unit unit
	);

	/** Convert natural units to arbitrary units. */
	template<typename Quantity>
	static double convertNaturalToArbitrary(
		double value,
		typename Quantity::Unit unit
	);

	/** Convert mass from derived units to base units. */
	static double convertMassDerivedToBase(double mass, MassUnit unit);

	/** Convert mass from base units to derived units. */
	static double convertMassBaseToDerived(double mass, MassUnit unit);

	/** Convert mass from derived units to natural units. */
	static double convertMassDerivedToNatural(double mass, MassUnit unit);

	/** Convert mass from natural units to derived units. */
	static double convertMassNaturalToDerived(double mass, MassUnit unit);

	/** Convert magnetic field from derived units to base units. */
	static double convertMagneticFieldDerivedToBase(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from base units to derived units. */
	static double convertMagneticFieldBaseToDerived(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from derived units to natural units. */
	static double convertMagneticFieldDerivedToNatural(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from natural units to derived units. */
	static double convertMagneticFieldNaturalToDerived(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert voltage from derived units to base units. */
	static double convertVoltageDerivedToBase(
		double voltage,
		VoltageUnit unit
	);

	/** Convert voltage from base units to derived units. */
	static double convertVoltageBaseToDerived(
		double voltage,
		VoltageUnit unit
	);

	/** Convert voltage from derived units to natural units. */
	static double convertVoltageDerivedToNatural(
		double voltage,
		VoltageUnit unit
	);

	/** Convert voltage from natural units to derived units. */
	static double convertVoltageNaturalToDerived(
		double voltage,
		VoltageUnit unit
	);

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
		std::pair<double, std::vector<std::pair<std::string, int>>>
	> constantsDefaultUnits;

	/** Physical constans in the current base units. */
	static std::map<std::string, double> constantsBaseUnits;

	/** Conversion factor from eV to J. */
	static double J_per_eV;

	/** Conversion factor from eV to J. */
	static double eV_per_J;

	/** Conversion factor from eVs^2/m^2 to kg. */
	static double kg_per_baseMass;

	/** Conversion factor from kg to eVs^2/m^2. */
	static double baseMass_per_kg;

	/** Conversion factor from eVs^2/m^2 to u. */
	static double u_per_baseMass;

	/** Conversion factor from u to eVs^2/m^2. */
	static double baseMass_per_u;

	/** Conversion factor from eVs/Cm^2 to T. */
	static double T_per_baseMagneticField;

	/** Conversion factor from T to eVs/Cm^2. */
	static double baseMagneticField_per_T;

	/** Conversion factor from V to eV/C. */
	static double V_per_baseVoltage;

	/** Conversion factor from eV/C to V. */
	static double baseVoltage_per_V;

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
	 *  Quatity names. */
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

	/** Returns the number of degrees in the given unit per degree in
	 *  default unit (K). */
/*	static double getTemperatureConversionFactor(
		Quantity::Temperature::Unit temperatureUnit
	);*/

	/** Returns the number of unit times in the given unit per unit time in
	 *  the default unit (s). */
//	static double getTimeConversionFactor(Quantity::Time::Unit timeUnit);

	/** Returns the number of unit lengths in the given unit per unit
	 *  length in the default unit (m). */
/*	static double getLengthConversionFactor(
		Quantity::Length::Unit lengthUnit
	);*/

	/** Returns the number of unit energies in the given unit per unit
	 *  energy in the default unit (eV). */
/*	static double getEnergyConversionFactor(
		Quantity::Energy::Unit energyUnit
	);*/

	/** Returns the number of unit charges in the given unit per unit
	 *  charge in the default unit (C). */
/*	static double getChargeConversionFactor(
		Quantity::Charge::Unit chargeUnit
	);*/

	/** Returns the number of unit counts in the the given unit per unit
	 *  count in the default unit (pcs). */
/*	static double getCountConversionFactor(
		Quantity::Count::Unit countUnit
	);*/

	/** Returns the number of unit masses in the input unit per unit mass
	 *  in the default unit (eVs^2/m^2). */
	static double getMassConversionFactor(MassUnit unit);

	/** Returns the amount of unit magnetic field strength in the input
	 *  unit per unit magnetic field strength in the default unit
	 *  (eVs/Cm^2). */
	static double getMagneticFieldConversionFactor(MagneticFieldUnit unit);

	/** Returns the amount of unit voltage in the input unit per unit
	 *  voltage in the default unit (eV/C). */
	static double getVoltageConversionFactor(VoltageUnit unit);

	/** Converts a string into a corresponding Unit. */
	template<typename Quantity>
	static typename Quantity::Unit getUnit(const std::string &unit);

	/** Static constructor. */
	static class StaticConstructor{
	public:
		StaticConstructor();
	} staticConstructor;
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
double UnitHandler::convertArbitraryToBase(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
	)/getConversionFactor<Quantity>(unit);
}

template<typename Quantity>
double UnitHandler::convertBaseToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
		unit
	)/getConversionFactor<Quantity>();
}

template<typename Quantity>
double UnitHandler::convertArbitraryToNatural(
	double value,
	typename Quantity::Unit unit
){
	return value*getConversionFactor<Quantity>(
	)/(getConversionFactor<Quantity>(unit)*getScale<Quantity>());
}

template<typename Quantity>
double UnitHandler::convertNaturalToArbitrary(
	double value,
	typename Quantity::Unit unit
){
	return value*getScale<Quantity>()*getConversionFactor<Quantity>(
		unit
	)/getConversionFactor<Quantity>();
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Charge>(){
	return Quantity::Charge::getConversionFactor(
		getUnit<Quantity::Charge>()
	);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Count>(){
	return Quantity::Count::getConversionFactor(
		getUnit<Quantity::Count>()
	);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Energy>(){
	return Quantity::Energy::getConversionFactor(
		getUnit<Quantity::Energy>()
	);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Length>(){
	return Quantity::Length::getConversionFactor(
		getUnit<Quantity::Length>()
	);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Temperature>(){
	return Quantity::Temperature::getConversionFactor(
		getUnit<Quantity::Temperature>()
	);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Time>(){
	return Quantity::Time::getConversionFactor(getUnit<Quantity::Time>());
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Charge>(
	typename Quantity::Charge::Unit unit
){
	return Quantity::Charge::getConversionFactor(unit);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Count>(
	typename Quantity::Count::Unit unit
){
	return Quantity::Count::getConversionFactor(unit);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Energy>(
	typename Quantity::Energy::Unit unit
){
	return Quantity::Energy::getConversionFactor(unit);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Length>(
	typename Quantity::Length::Unit unit
){
	return Quantity::Length::getConversionFactor(unit);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Temperature>(
	typename Quantity::Temperature::Unit unit
){
	return Quantity::Temperature::getConversionFactor(unit);
}

template<>
inline double UnitHandler::getConversionFactor<Quantity::Time>(
	typename Quantity::Time::Unit unit
){
	return Quantity::Time::getConversionFactor(unit);
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
