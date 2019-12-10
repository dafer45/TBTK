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
 *  @brief Handles conversions between different units
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_HANDLER
#define COM_DAFER45_TBTK_UNIT_HANDLER

#include "TBTKMacros.h"

#include <string>
#include <vector>
#include <cmath>

#ifdef M_E	//Avoid name clash with math.h macro M_E
	#define M_E_temp M_E
	#undef M_E
#endif

namespace TBTK{

/** !!! Note: Numerica values are picked from Wikipedia, and have not yet been
 *  checked agains an authoritative source. Some values that should be the same
 *  are even known to differ in some of the least significant digits. !!!
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
	enum class TemperatureUnit {kK, K, mK, uK, nK};

	/* Time units (base unit):<br/>
	 *	s - second<br/>
	 *	ms - millisecond<br/>
	 *	us - microsecond<br/>
	 *	ns - nanosecond<br/>
	 *	ps - picosecond<br/>
	 *	fs - femtosecond<br/>
	 *	as - attosecond */
	enum class TimeUnit {s, ms, us, ns, ps, fs, as};

	/** Length units (base unit):<br/>
	 *	m - meter<br/>
	 *	mm - millimeter<br/>
	 *	um - micrometer<br/>
	 *	nm - nanometer<br/>
	 *	pm - picometer<br/>
	 *	fm - femtometer<br/>
	 *	am - attometer<br/>
	 *	Ao - Angstrom */
	enum class LengthUnit{m, mm, um, nm, pm, fm, am, Ao};

	/** Energy units (base unit):<br/>
	 *	GeV - gigaelectron Volt<br/>
	 *	MeV - megaelectron Volt<br/>
	 *	keV - kiloelectron Volt<br/>
	 *	eV - electron Volt<br/>
	 *	meV - millielectron Volt<br/>
	 *	ueV - microelectron Volt<br/>
	 *	J - Joule */
	enum class EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};

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
	enum class ChargeUnit{
		kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e
	};

	/** Count unit (base unit):
	 *	pcs - pieces
	 *	mol - Mole */
	enum class CountUnit{pcs, mol};

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

	/** Get the Planck constant in the currently set base units. */
	static double getHbarB();

	/** Get the Planck constant in the currently set natural units. */
	static double getHbarN();

	/** Get the Boltzmann constant in the currently set base units. */
	static double getK_BB();

	/** Get the Boltzmann constant in the currently set natural units. */
	static double getK_BN();

	/** Get the elementary charge in the currently set base units. */
	static double getEB();

	/** Get the elementary charge in the currently set natural units. */
	static double getEN();

	/** Get the speed of light in the curently set base units. */
	static double getCB();

	/** Get the speed of light in the curently set natural units. */
	static double getCN();

	/** Get Avogadros number in the currently set base units. */
	static double getN_AB();

	/** Get Avogadros number in the currently set natural units. */
	static double getN_AN();

	/** Get the electron mass in the currently set base units. */
	static double getM_eB();

	/** Get the electron mass in the currently set natural units. */
	static double getM_eN();

	/** Get the proton mass in the currently set base units. */
	static double getM_pB();

	/** Get the proton mass in the currently set natural units. */
	static double getM_pN();

	/** Get the Bohr magneton in the currently set base units. */
	static double getMu_BB();

	/** Get the Bohr magneton in the currently set natural units. */
	static double getMu_BN();

	/** Get the nuclear magneton in the currently set base units. */
	static double getMu_nB();

	/** Get the nuclear magneton in the currently set natural units. */
	static double getMu_nN();

	/** Get the vacuum permeability in the currently set base units. */
	static double getMu_0B();

	/** Get the vacuum permeability in the currently set natural units. */
	static double getMu_0N();

	/** Get the vacuum permittivity in the currently set base units. */
	static double getEpsilon_0B();

	/** Get the vacuum permittivity in the currently set natural units. */
	static double getEpsilon_0N();

	/** Get the Bohr radius in the currently set base units. */
	static double getA_0B();

	/** Get the Bohr radius in the currently set natural units. */
	static double getA_0N();

	/** Set scales. */
	static void setScales(const std::vector<std::string> &scales);

	/** Convert temperature from natural units to base units. */
	static double convertTemperatureNtB(double temperature);

	/** Convert time to from natural units to base units. */
	static double convertTimeNtB(double time);

	/** Convert length from natural units to base units. */
	static double convertLengthNtB(double length);

	/** Convert energy from natural units to base units */
	static double convertEnergyNtB(double energy);

	/** Convert charge from natural units to base units. */
	static double convertChargeNtB(double charge);

	/** Conver counting from natural units to base units. */
	static double convertCountNtB(double count);

	/** Convert temperature from base units to natural units. */
	static double convertTemperatureBtN(double temperature);

	/** Convert time from base units to natural units. */
	static double convertTimeBtN(double time);

	/** Convert length from base units to natural units. */
	static double convertLengthBtN(double length);

	/** Convert energy from base units to natural units. */
	static double convertEnergyBtN(double energy);

	/** Convert charge from base units to natural units. */
	static double convertChargeBtN(double charge);

	/** Convert count from base units to natural units. */
	static double convertCountBtN(double count);

	/** Convert temperature from arbitrary units to base units. */
	static double convertTemperatureAtB(
		double temperature,
		TemperatureUnit unit
	);

	/** Convert time from arbitrary units to base units. */
	static double convertTimeAtB(
		double time,
		TimeUnit unit
	);

	/** Convert length from arbitrary units to base units. */
	static double convertLengthAtB(
		double length,
		LengthUnit unit
	);

	/** Convert energy from arbitrary units to base units. */
	static double convertEnergyAtB(
		double energy,
		EnergyUnit unit
	);

	/** Convert charge from arbitrary units to base units. */
	static double convertChargeAtB(
		double charge,
		ChargeUnit unit
	);

	/** Convert count from arbitrary units to base units. */
	static double convertCountAtB(
		double count,
		CountUnit unit
	);

	/** Convert temperature from base units to arbitrary units. */
	static double convertTemperatureBtA(
		double temperature,
		TemperatureUnit unit
	);

	/** Convert time from base units to arbitrary units. */
	static double convertTimeBtA(
		double time,
		TimeUnit unit
	);

	/** Convert length from base units to arbitrary units. */
	static double convertLengthBtA(
		double length,
		LengthUnit unit
	);

	/** Convert energy from base units to arbitrary units. */
	static double convertEnergyBtA(
		double energy,
		EnergyUnit unit
	);

	/** Convert charge from base units to arbitrary units. */
	static double convertChargeBtA(
		double charge,
		ChargeUnit unit
	);

	/** Convert count from base units to arbitrary units. */
	static double convertCountBtA(
		double count,
		CountUnit unit
	);

	/** Convert temperature from arbitrary units to natural units. */
	static double convertTemperatureAtN(
		double temperature,
		TemperatureUnit unit
	);

	/** Convert time from arbitrary units to natural units. */
	static double convertTimeAtN(
		double time,
		TimeUnit unit
	);

	/** Convert length from arbitrary units to natural units. */
	static double convertLengthAtN(
		double length,
		LengthUnit unit
	);

	/** Convert energy from arbitrary units to natural units. */
	static double convertEnergyAtN(
		double energy,
		EnergyUnit unit
	);

	/** Convert charge from arbitrary units to natural units. */
	static double convertChargeAtN(
		double charge,
		ChargeUnit unit
	);

	/** Convert count from arbitrary units to natural units. */
	static double convertCountAtN(
		double count,
		CountUnit unit
	);

	/** Convert temperature from natural units to arbitrary units. */
	static double convertTemperatureNtA(
		double temperature,
		TemperatureUnit unit
	);

	/** Convert time from natural units to arbitrary units. */
	static double convertTimeNtA(
		double time,
		TimeUnit unit
	);

	/** Convert length from natural units to arbitrary units. */
	static double convertLengthNtA(
		double length,
		LengthUnit unit
	);

	/** Convert energy from natural units to arbitrary units. */
	static double convertEnergyNtA(
		double energy,
		EnergyUnit unit
	);

	/** Convert charge from natural units to arbitrary units. */
	static double convertChargeNtA(
		double charge,
		ChargeUnit unit
	);

	/** Convert count from natural units to arbitrary units. */
	static double convertCountNtA(
		double count,
		CountUnit unit
	);

	/** Convert mass from derived units to base units. */
	static double convertMassDtB(double mass, MassUnit unit);

	/** Convert mass from base units to derived units. */
	static double convertMassBtD(double mass, MassUnit unit);

	/** Convert mass from derived units to natural units. */
	static double convertMassDtN(double mass, MassUnit unit);

	/** Convert mass from natural units to derived units. */
	static double convertMassNtD(double mass, MassUnit unit);

	/** Convert magnetic field from derived units to base units. */
	static double convertMagneticFieldDtB(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from base units to derived units. */
	static double convertMagneticFieldBtD(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from derived units to natural units. */
	static double convertMagneticFieldDtN(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert magnetic field from natural units to derived units. */
	static double convertMagneticFieldNtD(
		double field,
		MagneticFieldUnit unit
	);

	/** Convert voltage from derived units to base units. */
	static double convertVoltageDtB(double voltage, VoltageUnit unit);

	/** Convert voltage from base units to derived units. */
	static double convertVoltageBtD(double voltage, VoltageUnit unit);

	/** Convert voltage from derived units to natural units. */
	static double convertVoltageDtN(double voltage, VoltageUnit unit);

	/** Convert voltage from natural units to derived units. */
	static double convertVoltageNtD(double voltage, VoltageUnit unit);

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
	/** Planck constant in default units (eVs). */
	static constexpr double HBAR	= 6.582119514e-16;

	/** Boltzmann constant in default units (eV/K). */
	static constexpr double K_B	= 8.6173324e-5;

	/** Elementary charge in default units (C). */
	static constexpr double E	= 1.6021766208e-19;

	/** Speed of light in default units (m/s). */
	static constexpr double C	= 2.99792458e8;

	/** Avogadros number in default units (pcs). */
	static constexpr double N_A = 6.022140857e23;

	/** Electron mass in default units (eVs^2/m^2). */
	static constexpr double M_E	= 5.109989461e5/(C*C);

	/** Proton mass in default units (eVs^2/m^2). */
	static constexpr double M_P	= 9.38272046e8/(C*C);

	/** Bohr magneton in default units (Cm^2/s). */
	static constexpr double MU_B	= E*HBAR/(2.*M_E);

	/** Nuclear magneton in default units (Cm^2/s). */
	static constexpr double MU_N	= E*HBAR/(2*M_P);

	/** Vacuum permeability in default units (eVs^2/C^2m). */
	static constexpr double MU_0 = 4*M_PI*1e-7/1.602176565e-19;

	/** Vacuum permittivity in default units (C^2/eVm). */
	static constexpr double EPSILON_0 = 8.854187817620e-12*1.602176565e-19;

	/** Bohr radius in default units (m). */
	static constexpr double A_0 = 5.2917721092*1e-11;

	/** Planck constant in the currently set units. */
	static double hbar;

	/** Boltzmann constant in the currently set units. */
	static double k_B;

	/** Elementary charge in the currently set units. */
	static double e;

	/** Speed of light in the currently set units. */
	static double c;

	/** Avogadros number in the currently set units. */
	static double n_A;

	/** Electron mass in the currently set units. */
	static double m_e;

	/** Electron mass in the currently set units. */
	static double m_p;

	/** Bohr magneton in the currently set units. */
	static double mu_B;

	/** Nuclear magneton in the currently set units. */
	static double mu_n;

	/** Vacuum permeability in the currently set units. */
	static double mu_0;

	/** Vacuum permittivity in the currently set units. */
	static double epsilon_0;

	/** Bohr radius in the currently set units. */
	static double a_0;

	/** Conversion factor from eV to J. */
	static constexpr double J_per_eV	= 1.602176565e-19;

	/** Conversion factor from eV to J. */
	static constexpr double eV_per_J	= 1./J_per_eV;

	/** Conversion factor from eVs^2/m^2 to kg. */
	static constexpr double kg_per_baseMass = 1.602176565e-19;

	/** Conversion factor from kg to eVs^2/m^2. */
	static constexpr double baseMass_per_kg = 1./kg_per_baseMass;

	/** Conversion factor from eVs^2/m^2 to u. */
//	static constexpr double u_per_baseMass = 9.31494095e8/(C*C);
	static constexpr double u_per_baseMass = (C*C)/9.31494095e8;

	/** Conversion factor from u to eVs^2/m^2. */
	static constexpr double baseMass_per_u = 1./u_per_baseMass;

	/** Conversion factor from eVs/Cm^2 to T. */
	static constexpr double T_per_baseMagneticField = 1.602176565e-19;

	/** Conversion factor from T to eVs/Cm^2. */
	static constexpr double baseMagneticField_per_T = 1./T_per_baseMagneticField;

	/** Conversion factor from V to eV/C. */
	static constexpr double V_per_baseVoltage = 1.602176565e-19;

	/** Conversion factor from eV/C to V. */
	static constexpr double baseVoltage_per_V = 1./V_per_baseVoltage;

	/** Currently set temperature unit. */
	static TemperatureUnit temperatureUnit;

	/** Currently set time unit. */
	static TimeUnit timeUnit;

	/** Currently set length unit. */
	static LengthUnit lengthUnit;

	/** Currently set energy unit. */
	static EnergyUnit energyUnit;

	/** Currently set charge unit. */
	static ChargeUnit chargeUnit;

	/**Currently set count unit. */
	static CountUnit countUnit;

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
	static void setTemperatureUnit(TemperatureUnit unit);

	/** Set time unit. */
	static void setTimeUnit(TimeUnit unit);

	/** Set length unit. */
	static void setLengthUnit(LengthUnit unit);

	/** Set energy unit. */
	static void setEnergyUnit(EnergyUnit unit);

	/** Set charge unit. */
	static void setChargeUnit(ChargeUnit unit);

	/** Set counting unit. */
	static void setCountUnit(CountUnit unit);

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
	static void setTemperatureScale(double scale, TemperatureUnit unit);

	/** Set time scale. */
	static void setTimeScale(double scale, TimeUnit unit);

	/** Set length scale. */
	static void setLengthScale(double scale, LengthUnit unit);

	/** Set energy scale. */
	static void setEnergyScale(double scale, EnergyUnit unit);

	/** Set charge scale. */
	static void setChargeScale(double scale, ChargeUnit unit);

	/** Set count scale. */
	static void setCountScale(double scale, CountUnit unit);

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

	/** Update Planck constant. To be called at change of units. */
	static void updateHbar();

	/** Update Boltzmann constant. To be called at change of units. */
	static void updateK_B();

	/** Update elementary charge. To be called at change of units. */
	static void updateE();

	/** Update speed of light. To be called at change of units. */
	static void updateC();

	/** Update Avogadros number. To be called at change of units. */
	static void updateN_A();

	/** Update electron mass. To be called at change of units. */
	static void updateM_e();

	/** Update proton mass. To be called at change of units. */
	static void updateM_p();

	/** Update Bohr magneton. To be called at change of units. */
	static void updateMu_B();

	/** Update nuclear magneton. To be called at change of units. */
	static void updateMu_n();

	/** Update vacuum permeability. To be called at change of units. */
	static void updateMu_0();

	/** Update vacuum permittivity. To be called at change of units. */
	static void updateEpsilon_0();

	/** Update the Bohr radius. To be called at change of units. */
	static void updateA_0();

	/** Returns the number of degrees in the currently set unit per degree
	 *  in default unit (K). */
	static double getTemperatureConversionFactor();

	/** Returns the number of degrees in the given unit per degree in
	 *  default unit (K). */
	static double getTemperatureConversionFactor(
		TemperatureUnit temperatureUnit
	);

	/** Returns the number of unit times in the currently set unit per unit
	 * time in the default unit (s). */
	static double getTimeConversionFactor();

	/** Returns the number of unit times in the given unit per unit time in
	 *  the default unit (s). */
	static double getTimeConversionFactor(TimeUnit timeUnit);

	/** Returns the number of unit lengths in the currently set unit per
	 *  unit length in the default unit (m). */
	static double getLengthConversionFactor();

	/** Returns the number of unit lengths in the given unit per unit
	 *  length in the default unit (m). */
	static double getLengthConversionFactor(LengthUnit lengthUnit);

	/** Returns the number of unit energies in the currently set unit per
	 *  unit energy in the default unit (eV). */
	static double getEnergyConversionFactor();

	/** Returns the number of unit energies in the given unit per unit
	 *  energy in the default unit (eV). */
	static double getEnergyConversionFactor(EnergyUnit energyUnit);

	/** Returns the number of unit charges in the currently set unit per
	 *  unit charge in the default unit (C). */
	static double getChargeConversionFactor();

	/** Returns the number of unit charges in the given unit per unit
	 *  charge in the default unit (C). */
	static double getChargeConversionFactor(ChargeUnit chargeUnit);

	/** Returns the number of unit counts in the the currently set unit per
	 *  unit count in the default unit (pcs). */
	static double getCountConversionFactor();

	/** Returns the number of unit counts in the the given unit per unit
	 *  count in the default unit (pcs). */
	static double getCountConversionFactor(CountUnit countUnit);

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
	static TemperatureUnit getTemperatureUnit(std::string unit);

	/** Converts a string into a TimeUnit. */
	static TimeUnit getTimeUnit(std::string unit);

	/** Converts a string into a LengthUnit. */
	static LengthUnit getLengthUnit(std::string unit);

	/** Converts a string into a EnergyUnit. */
	static EnergyUnit getEnergyUnit(std::string unit);

	/** Converts a string into a ChargeUnit. */
	static ChargeUnit getChargeUnit(std::string unit);

	/** Converts a string into a CountUnit. */
	static CountUnit getCountUnit(std::string unit);
};

inline double UnitHandler::getHbarB(){
	return hbar;
}

inline double UnitHandler::getHbarN(){
	return hbar/(energyScale*timeScale);
}

inline double UnitHandler::getK_BB(){
	return k_B;
}

inline double UnitHandler::getK_BN(){
	return k_B*temperatureScale/energyScale;
}

inline double UnitHandler::getEB(){
	return e;
}

inline double UnitHandler::getEN(){
	return e/chargeScale;
}

inline double UnitHandler::getCB(){
	return c;
}

inline double UnitHandler::getCN(){
	return c*timeScale/lengthScale;
}

inline double UnitHandler::getN_AB(){
	return n_A;
}

inline double UnitHandler::getN_AN(){
	return n_A/countScale;
}

inline double UnitHandler::getM_eB(){
	return m_e;
}

inline double UnitHandler::getM_eN(){
	return m_e*lengthScale*lengthScale/(energyScale*timeScale*timeScale);
}

inline double UnitHandler::getM_pB(){
	return m_p;
}

inline double UnitHandler::getM_pN(){
	return m_p*lengthScale*lengthScale/(energyScale*timeScale*timeScale);
}

inline double UnitHandler::getMu_BB(){
	return mu_B;
}

inline double UnitHandler::getMu_BN(){
	return mu_B*timeScale/(chargeScale*lengthScale*lengthScale);
}

inline double UnitHandler::getMu_nB(){
	return mu_n;
}

inline double UnitHandler::getMu_nN(){
	return mu_n*timeScale/(chargeScale*lengthScale*lengthScale);
}

inline double UnitHandler::getMu_0B(){
	return mu_0;
}

inline double UnitHandler::getMu_0N(){
	return mu_0*chargeScale*chargeScale*lengthScale/(energyScale*timeScale*timeScale);
}

inline double UnitHandler::getEpsilon_0B(){
	return epsilon_0;
}

inline double UnitHandler::getEpsilon_0N(){
	return epsilon_0*energyScale*lengthScale/(chargeScale*chargeScale);
}

inline double UnitHandler::getA_0B(){
	return a_0;
}

inline double UnitHandler::getA_0N(){
	return a_0/lengthScale;
}

inline void UnitHandler::setTemperatureScale(double scale, TemperatureUnit unit){
	setTemperatureUnit(unit);
	setTemperatureScale(scale);
}

inline void UnitHandler::setTimeScale(double scale, TimeUnit unit){
	setTimeUnit(unit);
	setTimeScale(scale);
}

inline void UnitHandler::setLengthScale(double scale, LengthUnit unit){
	setLengthUnit(unit);
	setLengthScale(scale);
}

inline void UnitHandler::setEnergyScale(double scale, EnergyUnit unit){
	setEnergyUnit(unit);
	setEnergyScale(scale);
}

inline void UnitHandler::setChargeScale(double scale, ChargeUnit unit){
	setChargeUnit(unit);
	setChargeScale(scale);
}

inline void UnitHandler::setCountScale(double scale, CountUnit unit){
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
}

inline double UnitHandler::convertTemperatureNtB(double temperature){
	return temperature*temperatureScale;
}

inline double UnitHandler::convertTimeNtB(double time){
	return time*timeScale;
}

inline double UnitHandler::convertLengthNtB(double length){
	return length*lengthScale;
}

inline double UnitHandler::convertEnergyNtB(double energy){
	return energy*energyScale;
}

inline double UnitHandler::convertChargeNtB(double charge){
	return charge*chargeScale;
}

inline double UnitHandler::convertCountNtB(double count){
	return count*countScale;
}

inline double UnitHandler::convertTemperatureBtN(double temperature){
	return temperature/temperatureScale;
}

inline double UnitHandler::convertTimeBtN(double time){
	return time/timeScale;
}

inline double UnitHandler::convertLengthBtN(double length){
	return length/lengthScale;
}

inline double UnitHandler::convertEnergyBtN(double energy){
	return energy/energyScale;
}

inline double UnitHandler::convertChargeBtN(double charge){
	return charge/chargeScale;
}

inline double UnitHandler::convertCountBtN(double count){
	return count/countScale;
}

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
