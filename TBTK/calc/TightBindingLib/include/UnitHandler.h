/** @package TBTKcalc
 *  @file UnitHandler.h
 *  @brief Handles conversions between different units
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_HANDLER
#define COM_DAFER45_TBTK_UNIT_HANDLER

#include <string>

namespace TBTK{

class UnitHandler{
public:
	/** Temperature units:
	 *	K - Kelvin */
	enum TemperatureUnit {K};

	/* Time units:
	 *	s - second
	 *	ms - millisecond
	 *	us - microsecond
	 *	ns - nanosecond
	 *	ps - picosecond
	 *	fs - femtosecond
	 *	as - attosecond */
	enum TimeUnit{s, ms, us, ns, ps, fs, as};

	/** Distance units:
	 *	m - meter
	 *	mm - millimeter
	 *	um - micrometer
	 *	nm - nanometer
	 *	pm - picometer
	 *	fm - femtometer
	 *	am - attometer
	 *	Ao - Angstrom */
	enum LengthUnit{m, mm, um, nm, pm, fm, am, Ao};

	/** Energy units:
	 *	GeV - gigaelectron Volt
	 *	MeV - megaelectron Volt
	 *	keV - kiloelectron Volt
	 *	eV - electron Volt
	 *	meV - millielectron Volt
	 *	ueV - microelectron Volt
	 *	J - Joule */
	enum EnergyUnit{GeV, MeV, keV, eV, meV, ueV, J};

	/** Planck constant in the currently set units. */
	static double hbar;

	/** Boltzmann constant in the currently set units. */
	static double k_b;

	/** Set temperature unit. */
	static void setTemperatureUnit(TemperatureUnit unit);

	/** Set time unit. */
	static void setTimeUnit(TimeUnit unit);

	/** Set length unit. */
	static void setLengthUnit(LengthUnit unit);

	/** Set energy unit. */
	static void setEnergyUnit(EnergyUnit unit);

	/** Set temperature scale. */
	static void setTemperatureScale(double scale);

	/** Set time scale. */
	static void setTimeScale(double scale);

	/** Set length scale. */
	static void setLengthScale(double scale);

	/** Set energy scale. */
	static void setEnergyScale(double scale);

	/** Get temperature unit string
	 *
	 * @return string representation of the currently set temperature unit.
	 */
	static std::string getTemperatureUnitString();

	/** Get time unit string
	 *
	 * @return string representation of the currently set time unit. */
	static std::string getTimeUnitString();

	/** Get length unit string
	 *
	 * @return string representation of the currently set length unit. */
	static std::string getLengthUnitString();

	/** Get energy unit string
	 *
	 * @return string representation of the currently set energy unit. */
	static std::string getEnergyUnitString();

	/** Get Planck constant unit string
	 *
	 * @retrun string representation of the unit for the Planck constant.
	 */
	static std::string getHBARUnitString();

	/** Get Boltzmann constant unit string.
	 *
	 * @return string representation of the unit for the Boltzmann
	 * constant. */
	static std::string getK_BUnitString();
private:
	/** Planck constant in default units (eVs). */
	static constexpr double HBAR	= 6.582119514e-16;

	/** Boltzmann constant in default units (eV/K). */
	static constexpr double K_B	= 8.6173324e-5;

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

	/** Currently set temperature scale. */
	static double temperatureScale;

	/** Currently set time scale. */
	static double timeScale;

	/** Currently set length scale. */
	static double lengthScale;

	/** Currently set energy scale. */
	static double energyScale;

	/** Update Planck constant. To be called at change of units. */
	static void updateHbar();

	/** Update Boltzmann constant. To be called at change of units. */
	static void updateK_b();

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
};

};

#endif
