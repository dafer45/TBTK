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

/** The UnitHandler handles conversion between the 'natural units' used in the
 *  calculations and standard units. The base quantities are Temperature, time,
 *  length, energy, and charge, and the default units are Kelvin (K), second
 *  (s), meter (m), electron Volt (eV), and Coulomb (C).
 *
 *  The current units can be changed with setXXXUnit(), where \f XXX \in
 *  \{Temperature, Time, Length, Energy, Charge\}\f, and subsequent calls will
 *  take input paramaters and return output parameters in corresponding units.
 *
 *  The functions setXXXScale() can be used to specify the scale of the 'ruler'
 *  with which parameters in the calculation are meassured. For example:
 *
 *  UnitHandler::setEnergyUnits(UnitHandler::EnergyUnit:meV);
 *  UnitHandler::setEnergyScale(1.47);
 *
 *  Here the energy unit is first set to meV, and the scale subsequently set to
 *  1.47meV. An energy parameter with value 1.0 will therefore be interpreted as
 *  having the physical value 1.47meV whenever a multiplication with a physical
 *  constant is required to generate a unitless number.
 *
 *  Note that the order of the above calls are important. Setting the energy
 *  scale to 1.47 while the default energy unit eV is used and then changing to
 *  meV as units will result in the scale becoming 1.47eV = 1470meV. */
class UnitHandler{
public:
	/** Temperature units:
	 *	K - Kelvin */
	enum class TemperatureUnit {K};

	/* Time units:
	 *	s - second
	 *	ms - millisecond
	 *	us - microsecond
	 *	ns - nanosecond
	 *	ps - picosecond
	 *	fs - femtosecond
	 *	as - attosecond */
	enum class TimeUnit {s, ms, us, ns, ps, fs, as};

	/** Distance units:
	 *	m - meter
	 *	mm - millimeter
	 *	um - micrometer
	 *	nm - nanometer
	 *	pm - picometer
	 *	fm - femtometer
	 *	am - attometer
	 *	Ao - Angstrom */
	enum class LengthUnit{m, mm, um, nm, pm, fm, am, Ao};

	/** Energy units:
	 *	GeV - gigaelectron Volt
	 *	MeV - megaelectron Volt
	 *	keV - kiloelectron Volt
	 *	eV - electron Volt
	 *	meV - millielectron Volt
	 *	ueV - microelectron Volt
	 *	J - Joule */
	enum class EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};

	/** Charge units:
	 *	C - Coulomb */
	enum class ChargeUnit{C};

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

	/** Convert temperature to the currently set units, assuming that the
	 *  input temperature is meassured with the scale set by
	 *  setTemperatureScale(). */
	static double convertTemperature(double temperature);

	/** Convert time to the currently set units, assuming that the input
	 *  time is meassured with the scale set by setTimeScale(). */
	static double convertTime(double time);

	/** Convert length to the currently set units, assuming that the input
	 *  length is meassured with the scale set by setLengthScale(). */
	static double convertLength(double length);

	/** Convert energy to the currently set units, assuming that the input
	 *  energy is meassured with the scale set by setEnergyScale(). */
	static double convertEnergy(double energy);

	/** Convert charge to the currently set units, assuming that the input
	 *  charge is meassured with the scale set by setEnergyScale(). */
	static double convertCharge(double charge);

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

	/** Bohr magneton in default units (Cm^2/s^2). */
	static constexpr double MU_B	= E*HBAR/(2.*M_E);

	/** Nuclear magneton in default units (Cm^2/s^2). */
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

inline double UnitHandler::convertTemperature(double temperature){
	return temperature*temperatureScale;
}

inline double UnitHandler::convertTime(double time){
	return time*timeScale;
}

inline double UnitHandler::convertLength(double length){
	return length*lengthScale;
}

inline double UnitHandler::convertEnergy(double energy){
	return energy*energyScale;
}

inline double UnitHandler::convertCharge(double charge){
	return charge*chargeScale;
}

};

#ifdef M_E_temp	//Avoid name clash with math.h macro M_E
	#define M_E M_E_temp
	#undef M_E_temp
#endif

#endif
