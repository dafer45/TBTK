/** @package TBTKcalc
 *  @file UnitHandler.h
 *  @brief Handles conversions between different units
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_HANDLER
#define COM_DAFER45_TBTK_UNIT_HANDLER

#include <string>

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
 *  Temperature, time, length, energy, and charge, and the default units are
 *  Kelvin (K), second (s), meter (m), electron Volt (eV), and Coulomb (C).
 *
 *  The current base units can be changed with setXXXUnit(), where \f$ XXX \in
 *  \{Temperature, Time, Length, Energy, Charge\}\f$, and subsequent calls will
 *  take input paramaters and return output parameters in corresponding units.
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
	/** Temperature units (base unit):
	 *	K - Kelvin */
	enum class TemperatureUnit {K};

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
	 *	C - Coulomb */
	enum class ChargeUnit{C};

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
	enum class MagneticFieldUnit{MT, kT, T, mT, uT, nT, GG, MG, kG, G, mG, uG};

	/** Get the Planck constant in the currently set units. */
	static double getHbar();

	/** Get the Boltzmann constant in the currently set units. */
	static double getK_b();

	/** Get the elementary charge in the currently set units. */
	static double getE();

	/** Get the speed of light in the curently set units. */
	static double getC();

	/** Get the electron mass in the currently set units. */
	static double getM_e();

	/** Get the proton mass in the currently set units. */
	static double getM_p();

	/** Get the Bohr magneton in the currently set units. */
	static double getMu_b();

	/** Get the nuclear magneton in the currently set units. */
	static double getMu_n();

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

	/** Convert mass from derived units to base units. */
	static double convertMassDtB(double mass, MassUnit unit);

	/** Convert mass from base units to derived units. */
	static double convertMassBtD(double mass, MassUnit unit);

	/** Convert mass from derived units to natural units. */
	static double convertMassDtN(double mass, MassUnit unit);

	/** Convert mass from natural units to derived units. */
	static double convertMassNtD(double mass, MassUnit unit);

	/** Convert magnetic field from derived units to base units. */
	static double convertMagneticFieldDtB(double field, MagneticFieldUnit unit);

	/** Convert magnetic field from base units to derived units. */
	static double convertMagneticFieldBtD(double field, MagneticFieldUnit unit);

	/** Convert magnetic field from derived units to natural units. */
	static double convertMagneticFieldDtN(double field, MagneticFieldUnit unit);

	/** Convert magnetic field from natural units to derived units. */
	static double convertMagneticFieldNtD(double field, MagneticFieldUnit unit);

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
	static std::string getMu_bUnitString();

	/** Get nuclear magneton unit string.
	 *
	 *  @return string representation of the unit for the nuclear magneton. */
	static std::string getMu_nUnitString();
private:
	/** Planck constant in default units (eVs). */
	static constexpr double HBAR	= 6.582119514e-16;

	/** Boltzmann constant in default units (eV/K). */
	static constexpr double K_B	= 8.6173324e-5;

	/** Elementary charge in default units (C). */
	static constexpr double E	= 1.6021766208e-19;

	/** Speed of light in default units (m/s). */
	static constexpr double C	= 2.99792458e8;

	/** Electron mass in default units (eVs^2/m^2). */
	static constexpr double M_E	= 5.109989461e5/(C*C);

	/** Proton mass in default units (eVs^2/m^2). */
	static constexpr double M_P	= 9.38272046e8/(C*C);

	/** Bohr magneton in default units (Cm^2/s). */
	static constexpr double MU_B	= E*HBAR/(2.*M_E);

	/** Nuclear magneton in default units (Cm^2/s). */
	static constexpr double MU_N	= E*HBAR/(2*M_P);

	/** Planck constant in the currently set units. */
	static double hbar;

	/** Boltzmann constant in the currently set units. */
	static double k_b;

	/** Elementary charge in the currently set units. */
	static double e;

	/** Speed of light in the currently set units. */
	static double c;

	/** Electron mass in the currently set units. */
	static double m_e;

	/** Electron mass in the currently set units. */
	static double m_p;

	/** Bohr magneton in the currently set units. */
	static double mu_b;

	/** Nuclear magneton in the currently set units. */
	static double mu_n;

	/** Conversion factor from eV to J. */
	static constexpr double J_per_eV	= 1.602176565e-19;

	/** Conversion factor from eV to J. */
	static constexpr double eV_per_J	= 1./J_per_eV;

	/** Conversion factor from eVs^2/m^2 to kg. */
	static constexpr double kg_per_baseMass = 1.602176565e-19;

	/** Conversion factor from kg to eVs^2/m^2. */
	static constexpr double baseMass_per_kg = 1./kg_per_baseMass;

	/** Conversion factor from eVs^2/m^2 to u. */
	static constexpr double u_per_baseMass = 9.31494095e8/(C*C);

	/** Conversion factor from u to eVs^2/m^2. */
	static constexpr double baseMass_per_u = 1./u_per_baseMass;

	/** Conversion factor from eVs/Cm^2 to T. */
	static constexpr double T_per_baseMagneticField = 1.602176565e-19;

	/** Conversion factor from T to eVs/Cm^2. */
	static constexpr double baseMagneticField_per_T = 1./T_per_baseMagneticField;

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

	/** Update Planck constant. To be called at change of units. */
	static void updateHbar();

	/** Update Boltzmann constant. To be called at change of units. */
	static void updateK_b();

	/** Update elementary charge. To be called at change of units. */
	static void updateE();

	/** Update speed of light. To be called at change of units. */
	static void updateC();

	/** Update electron mass. To be called at change of units. */
	static void updateM_e();

	/** Update proton mass. To be called at change of units. */
	static void updateM_p();

	/** Update Bohr magneton. To be called at change of units. */
	static void updateMu_b();

	/** Update nuclear magneton. To be called at change of units. */
	static void updateMu_n();

	/** Returns the number of degrees in the currently set unit per degree
	 *  in default unit (K). */
	static double getTemperatureConversionFactor();

	/** Returns the number of unit times in the currently set unit per unit
	 * time in the default unit (s). */
	static double getTimeConversionFactor();

	/** Returns the number of unit lengths in the currently set unit per
	 *  unit length in the default unit (m). */
	static double getLengthConversionFactor();

	/** Returns the number of unit energies in the currently set unit per
	 * unit energy in the default unit (eV). */
	static double getEnergyConversionFactor();

	/** Returns the number of unit charges in the currently set unit per
	 * unit charge in the default unit (C). */
	static double getChargeConversionFactor();

	/** Returns the number of unit masses in the input unit per unit mass
	 *  in the default unit (eVs^2/m^2). */
	static double getMassConversionFactor(MassUnit unit);

	/** Returns the amount of unit magnetic field strength in the input
	 *  unit per unit magnetic field strength in the default unit
	 *  (eVs/Cm^2). */
	static double getMagneticFieldConversionFactor(MagneticFieldUnit unit);
};

inline double UnitHandler::getHbar(){
	return hbar;
}

inline double UnitHandler::getK_b(){
	return k_b;
}

inline double UnitHandler::getE(){
	return e;
}

inline double UnitHandler::getC(){
	return c;
}

inline double UnitHandler::getM_e(){
	return m_e;
}

inline double UnitHandler::getM_p(){
	return m_p;
}

inline double UnitHandler::getMu_b(){
	return mu_b;
}

inline double UnitHandler::getMu_n(){
	return mu_n;
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

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
