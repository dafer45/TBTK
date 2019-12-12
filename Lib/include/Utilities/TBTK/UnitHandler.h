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
	/** Temperature units (base unit):<br/>
	 *	kK - kilokelvin<br/>
	 *	K - Kelvin<br/>
	 *	mK - millikelvin<br/>
	 *	uK - microkelvin<br/>
	 *	nK - nanokelvin*/
//	enum class TemperatureUnit {kK, K, mK, uK, nK};

	/* Time units (base unit):<br/>
	 *	s - second<br/>
	 *	ms - millisecond<br/>
	 *	us - microsecond<br/>
	 *	ns - nanosecond<br/>
	 *	ps - picosecond<br/>
	 *	fs - femtosecond<br/>
	 *	as - attosecond */
//	enum class TimeUnit {s, ms, us, ns, ps, fs, as};

	/** Length units (base unit):<br/>
	 *	m - meter<br/>
	 *	mm - millimeter<br/>
	 *	um - micrometer<br/>
	 *	nm - nanometer<br/>
	 *	pm - picometer<br/>
	 *	fm - femtometer<br/>
	 *	am - attometer<br/>
	 *	Ao - Angstrom */
//	enum class LengthUnit{m, mm, um, nm, pm, fm, am, Ao};

	/** Energy units (base unit):<br/>
	 *	GeV - gigaelectron Volt<br/>
	 *	MeV - megaelectron Volt<br/>
	 *	keV - kiloelectron Volt<br/>
	 *	eV - electron Volt<br/>
	 *	meV - millielectron Volt<br/>
	 *	ueV - microelectron Volt<br/>
	 *	J - Joule */
//	enum class EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};

	/** Charge units (base unit):<br/>
	 *	kC - kilocoulomb<br/>
	 *	C - Coulomb<br/>
	 *	mC - millicoulomb<br/>
	 *	uC - microcoulomb<br/>
	 *	nC - nanocoulomb<br/>
	 *	pC - picocoulomb<br/>
	 *	fC - femtocoulomb<br/>
	 *	aC - attocoulomb<br/>
	 *	Te - terrae<br/>
	 *	Ge - gigae<br/>
	 *	Me - megae<br/>
	 *	ke - kiloe<br/>
	 *	e - e (elementary charge) */
/*	enum class ChargeUnit{
		kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e
	};*/

	/** Count unit (base unit):
	 *	pcs - pieces
	 *	mol - Mole */
//	enum class CountUnit{pcs, mol};

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

	/** Convert temperature from natural units to base units. */
//	static double convertTemperatureNaturalToBase(double temperature);

	/** Convert time to from natural units to base units. */
//	static double convertTimeNaturalToBase(double time);

	/** Convert length from natural units to base units. */
//	static double convertLengthNaturalToBase(double length);

	/** Convert energy from natural units to base units */
//	static double convertEnergyNaturalToBase(double energy);

	/** Convert charge from natural units to base units. */
//	static double convertChargeNaturalToBase(double charge);

	/** Conver counting from natural units to base units. */
//	static double convertCountNaturalToBase(double count);

	/** Convert from base units to natural units. */
	template<typename Quantity>
	static double convertBaseToNatural(double value);

	/** Convert temperature from base units to natural units. */
//	static double convertTemperatureBaseToNatural(double temperature);

	/** Convert time from base units to natural units. */
//	static double convertTimeBaseToNatural(double time);

	/** Convert length from base units to natural units. */
//	static double convertLengthBaseToNatural(double length);

	/** Convert energy from base units to natural units. */
//	static double convertEnergyBaseToNatural(double energy);

	/** Convert charge from base units to natural units. */
//	static double convertChargeBaseToNatural(double charge);

	/** Convert count from base units to natural units. */
//	static double convertCountBaseToNatural(double count);

	/** Convert temperature from arbitrary units to base units. */
	static double convertTemperatureArbitraryToBase(
		double temperature,
		Quantity::Temperature::Unit unit
	);

	/** Convert time from arbitrary units to base units. */
	static double convertTimeArbitraryToBase(
		double time,
		Quantity::Time::Unit unit
	);

	/** Convert length from arbitrary units to base units. */
	static double convertLengthArbitraryToBase(
		double length,
		Quantity::Length::Unit unit
	);

	/** Convert energy from arbitrary units to base units. */
	static double convertEnergyArbitraryToBase(
		double energy,
		Quantity::Energy::Unit unit
	);

	/** Convert charge from arbitrary units to base units. */
	static double convertChargeArbitraryToBase(
		double charge,
		Quantity::Charge::Unit unit
	);

	/** Convert count from arbitrary units to base units. */
	static double convertCountArbitraryToBase(
		double count,
		Quantity::Count::Unit unit
	);

	/** Convert temperature from base units to arbitrary units. */
	static double convertTemperatureBaseToArbitrary(
		double temperature,
		Quantity::Temperature::Unit unit
	);

	/** Convert time from base units to arbitrary units. */
	static double convertTimeBaseToArbitrary(
		double time,
		Quantity::Time::Unit unit
	);

	/** Convert length from base units to arbitrary units. */
	static double convertLengthBaseToArbitrary(
		double length,
		Quantity::Length::Unit unit
	);

	/** Convert energy from base units to arbitrary units. */
	static double convertEnergyBaseToArbitrary(
		double energy,
		Quantity::Energy::Unit unit
	);

	/** Convert charge from base units to arbitrary units. */
	static double convertChargeBaseToArbitrary(
		double charge,
		Quantity::Charge::Unit unit
	);

	/** Convert count from base units to arbitrary units. */
	static double convertCountBaseToArbitrary(
		double count,
		Quantity::Count::Unit unit
	);

	/** Convert temperature from arbitrary units to natural units. */
	static double convertTemperatureArbitraryToNatural(
		double temperature,
		Quantity::Temperature::Unit unit
	);

	/** Convert time from arbitrary units to natural units. */
	static double convertTimeArbitraryToNatural(
		double time,
		Quantity::Time::Unit unit
	);

	/** Convert length from arbitrary units to natural units. */
	static double convertLengthArbitraryToNatural(
		double length,
		Quantity::Length::Unit unit
	);

	/** Convert energy from arbitrary units to natural units. */
	static double convertEnergyArbitraryToNatural(
		double energy,
		Quantity::Energy::Unit unit
	);

	/** Convert charge from arbitrary units to natural units. */
	static double convertChargeArbitraryToNatural(
		double charge,
		Quantity::Charge::Unit unit
	);

	/** Convert count from arbitrary units to natural units. */
	static double convertCountArbitraryToNatural(
		double count,
		Quantity::Count::Unit unit
	);

	/** Convert temperature from natural units to arbitrary units. */
	static double convertTemperatureNaturalToArbitrary(
		double temperature,
		Quantity::Temperature::Unit unit
	);

	/** Convert time from natural units to arbitrary units. */
	static double convertTimeNaturalToArbitrary(
		double time,
		Quantity::Time::Unit unit
	);

	/** Convert length from natural units to arbitrary units. */
	static double convertLengthNaturalToArbitrary(
		double length,
		Quantity::Length::Unit unit
	);

	/** Convert energy from natural units to arbitrary units. */
	static double convertEnergyNaturalToArbitrary(
		double energy,
		Quantity::Energy::Unit unit
	);

	/** Convert charge from natural units to arbitrary units. */
	static double convertChargeNaturalToArbitrary(
		double charge,
		Quantity::Charge::Unit unit
	);

	/** Convert count from natural units to arbitrary units. */
	static double convertCountNaturalToArbitrary(
		double count,
		Quantity::Count::Unit unit
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

	/** Get temperature unit string
	 *
	 *  @return string representation of the currently set temperature unit.
	 */
	static std::string getTemperatureUnitString();

	/** Get time unit string
	 *
	 *  @return string representation of the currently set time unit. */
	static std::string getTimeUnitString();

	/** Get length unit string
	 *
	 *  @return string representation of the currently set length unit. */
	static std::string getLengthUnitString();

	/** Get energy unit string
	 *
	 *  @return string representation of the currently set energy unit. */
	static std::string getEnergyUnitString();

	/** Get charge unit string
	 *
	 *  @return string representation of the currently set charge unit. */
	static std::string getChargeUnitString();

	/** Get count unit string
	 *
	 * @return string representation of the current set count unit. */
	static std::string getCountUnitString();

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

	/** Currently set temperature unit. */
	static Quantity::Temperature::Unit temperatureUnit;

	/** Currently set time unit. */
	static Quantity::Time::Unit timeUnit;

	/** Currently set length unit. */
	static Quantity::Length::Unit lengthUnit;

	/** Currently set energy unit. */
	static Quantity::Energy::Unit energyUnit;

	/** Currently set charge unit. */
	static Quantity::Charge::Unit chargeUnit;

	/**Currently set count unit. */
	static Quantity::Count::Unit countUnit;

	/** Currently set temperature scale. */
	static double temperatureScale;

	/** Currently set time scale. */
	static double timeScale;

	/** Currently set length scale. */
	static double lengthScale;

	/** Currently set energy scale. */
	static double energyScale;

	/** Currently set charge scale. */
	static double chargeScale;

	/** Currently set count scale. */
	static double countScale;

	/** Set temperature unit. */
	static void setTemperatureUnit(Quantity::Temperature::Unit unit);

	/** Set time unit. */
	static void setTimeUnit(Quantity::Time::Unit unit);

	/** Set length unit. */
	static void setLengthUnit(Quantity::Length::Unit unit);

	/** Set energy unit. */
	static void setEnergyUnit(Quantity::Energy::Unit unit);

	/** Set charge unit. */
	static void setChargeUnit(Quantity::Charge::Unit unit);

	/** Set counting unit. */
	static void setCountUnit(Quantity::Count::Unit unit);

	/** Set temperature scale. */
	static void setTemperatureScale(double scale);

	/** Set time scale. */
	static void setTimeScale(double scale);

	/** Set length scale. */
	static void setLengthScale(double scale);

	/** Set energy scale. */
	static void setEnergyScale(double scale);

	/** Set charge scale. */
	static void setChargeScale(double scale);

	/** Set count unit. */
	static void setCountScale(double scale);

	/** Set temperature scale. */
	static void setTemperatureScale(
		double scale,
		Quantity::Temperature::Unit unit
	);

	/** Set time scale. */
	static void setTimeScale(double scale, Quantity::Time::Unit unit);

	/** Set length scale. */
	static void setLengthScale(double scale, Quantity::Length::Unit unit);

	/** Set energy scale. */
	static void setEnergyScale(double scale, Quantity::Energy::Unit unit);

	/** Set charge scale. */
	static void setChargeScale(double scale, Quantity::Charge::Unit unit);

	/** Set count scale. */
	static void setCountScale(double scale, Quantity::Count::Unit unit);

	/** Set temperature scale. */
	static void setTemperatureScale(std::string scale);

	/** Set time scale. */
	static void setTimeScale(std::string scale);

	/** Set length scale. */
	static void setLengthScale(std::string scale);

	/** Set energy scale. */
	static void setEnergyScale(std::string scale);

	/** Set charge scale. */
	static void setChargeScale(std::string scale);

	/** Set count scale. */
	static void setCountScale(std::string scale);

	/** Get the scale factor for the corresponding quantity. */
	template<typename Quantity>
	static double getScaleFactor();

	/** Update contants to reflect the current base units. */
	static void updateConstants();

	/** Returns the number of degrees in the currently set unit per degree
	 *  in default unit (K). */
	static double getTemperatureConversionFactor();

	/** Returns the number of degrees in the given unit per degree in
	 *  default unit (K). */
	static double getTemperatureConversionFactor(
		Quantity::Temperature::Unit temperatureUnit
	);

	/** Returns the number of unit times in the currently set unit per unit
	 * time in the default unit (s). */
	static double getTimeConversionFactor();

	/** Returns the number of unit times in the given unit per unit time in
	 *  the default unit (s). */
	static double getTimeConversionFactor(Quantity::Time::Unit timeUnit);

	/** Returns the number of unit lengths in the currently set unit per
	 *  unit length in the default unit (m). */
	static double getLengthConversionFactor();

	/** Returns the number of unit lengths in the given unit per unit
	 *  length in the default unit (m). */
	static double getLengthConversionFactor(
		Quantity::Length::Unit lengthUnit
	);

	/** Returns the number of unit energies in the currently set unit per
	 *  unit energy in the default unit (eV). */
	static double getEnergyConversionFactor();

	/** Returns the number of unit energies in the given unit per unit
	 *  energy in the default unit (eV). */
	static double getEnergyConversionFactor(
		Quantity::Energy::Unit energyUnit
	);

	/** Returns the number of unit charges in the currently set unit per
	 *  unit charge in the default unit (C). */
	static double getChargeConversionFactor();

	/** Returns the number of unit charges in the given unit per unit
	 *  charge in the default unit (C). */
	static double getChargeConversionFactor(
		Quantity::Charge::Unit chargeUnit
	);

	/** Returns the number of unit counts in the the currently set unit per
	 *  unit count in the default unit (pcs). */
	static double getCountConversionFactor();

	/** Returns the number of unit counts in the the given unit per unit
	 *  count in the default unit (pcs). */
	static double getCountConversionFactor(
		Quantity::Count::Unit countUnit
	);

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

	/** Converts a string into a TemperatureUnit. */
	static Quantity::Temperature::Unit getTemperatureUnit(std::string unit);

	/** Converts a string into a TimeUnit. */
	static Quantity::Time::Unit getTimeUnit(std::string unit);

	/** Converts a string into a LengthUnit. */
	static Quantity::Length::Unit getLengthUnit(std::string unit);

	/** Converts a string into a EnergyUnit. */
	static Quantity::Energy::Unit getEnergyUnit(std::string unit);

	/** Converts a string into a ChargeUnit. */
	static Quantity::Charge::Unit getChargeUnit(std::string unit);

	/** Converts a string into a CountUnit. */
	static Quantity::Count::Unit getCountUnit(std::string unit);

	/** Static constructor. */
	static class StaticConstructor{
	public:
		StaticConstructor();
	} staticConstructor;
};

inline void UnitHandler::setTemperatureScale(
	double scale,
	Quantity::Temperature::Unit unit
){
	setTemperatureUnit(unit);
	setTemperatureScale(scale);
}

inline void UnitHandler::setTimeScale(double scale, Quantity::Time::Unit unit){
	setTimeUnit(unit);
	setTimeScale(scale);
}

inline void UnitHandler::setLengthScale(
	double scale,
	Quantity::Length::Unit unit
){
	setLengthUnit(unit);
	setLengthScale(scale);
}

inline void UnitHandler::setEnergyScale(
	double scale,
	Quantity::Energy::Unit unit
){
	setEnergyUnit(unit);
	setEnergyScale(scale);
}

inline void UnitHandler::setChargeScale(
	double scale,
	Quantity::Charge::Unit unit
){
	setChargeUnit(unit);
	setChargeScale(scale);
}

inline void UnitHandler::setCountScale(
	double scale,
	Quantity::Count::Unit unit
){
	setCountUnit(unit);
	setCountScale(scale);
}

inline void UnitHandler::setScales(const std::vector<std::string> &scales){
	TBTKAssert(
		scales.size() == 6,
		"UnitHandler::setScales()",
		"'scales' must contain six strings.",
		""
	);

	setChargeScale(scales[0]);
	setCountScale(scales[1]);
	setEnergyScale(scales[2]);
	setLengthScale(scales[3]);
	setTemperatureScale(scales[4]);
	setTimeScale(scales[5]);
	updateConstants();
}

template<typename Quantity>
double UnitHandler::convertNaturalToBase(double value){
	return value*getScaleFactor<Quantity>();
}

/*inline double UnitHandler::convertTemperatureNaturalToBase(double temperature){
	return temperature*temperatureScale;
}

inline double UnitHandler::convertTimeNaturalToBase(double time){
	return time*timeScale;
}

inline double UnitHandler::convertLengthNaturalToBase(double length){
	return length*lengthScale;
}

inline double UnitHandler::convertEnergyNaturalToBase(double energy){
	return energy*energyScale;
}

inline double UnitHandler::convertChargeNaturalToBase(double charge){
	return charge*chargeScale;
}

inline double UnitHandler::convertCountNaturalToBase(double count){
	return count*countScale;
}*/

template<typename Quantity>
double UnitHandler::convertBaseToNatural(double value){
	return value/getScaleFactor<Quantity>();
}

/*inline double UnitHandler::convertTemperatureBaseToNatural(double temperature){
	return temperature/temperatureScale;
}

inline double UnitHandler::convertTimeBaseToNatural(double time){
	return time/timeScale;
}

inline double UnitHandler::convertLengthBaseToNatural(double length){
	return length/lengthScale;
}

inline double UnitHandler::convertEnergyBaseToNatural(double energy){
	return energy/energyScale;
}

inline double UnitHandler::convertChargeBaseToNatural(double charge){
	return charge/chargeScale;
}

inline double UnitHandler::convertCountBaseToNatural(double count){
	return count/countScale;
}*/

template<>
inline double UnitHandler::getScaleFactor<Quantity::Charge>(){
	return chargeScale;
}

template<>
inline double UnitHandler::getScaleFactor<Quantity::Count>(){
	return countScale;
}

template<>
inline double UnitHandler::getScaleFactor<Quantity::Energy>(){
	return energyScale;
}

template<>
inline double UnitHandler::getScaleFactor<Quantity::Length>(){
	return lengthScale;
}

template<>
inline double UnitHandler::getScaleFactor<Quantity::Temperature>(){
	return temperatureScale;
}

template<>
inline double UnitHandler::getScaleFactor<Quantity::Time>(){
	return timeScale;
}

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
