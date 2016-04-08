/** @package TBTKcalc
 *  @file UnitHandler.h
 *  @brief Handles conversions between different units
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_UNIT_HANDLER
#define COM_DAFER45_TBTK_UNIT_HANDLER

namespace TBTK{

class UnitHandler{
public:
	/** Units:
	 *	Temperature:
	 *		K - Kelvin
	 *	Time:
	 *		s - second
	 *		ms - millisecond
	 *		us - microsecond
	 *		ns - nanosecond
	 *		ps - picosecond
	 *		fs - femtosecond
	 *		as - attosecond
	 *	Distance:
	 *		m - meter
	 *		mm - millimeter
	 *		um - micrometer
	 *		nm - nanometer
	 *		pm - picometer
	 *		fm - femtometer
	 *		am - attometer
	 *		Ao - Angstrom
	 *	Energy:
	 *		GeV - giga electron Volt
	 *		MeV - mega electron Volt
	 *		keV - kilo electron Volt
	 *		eV - electron Volt
	 *		meV - milli electron Volt
	 *		ueV - micro electron Volt
	 */
	enum {K, s, ms, us, ns, ps, fs, as, m, mm, um, nm, pm, fm, am, GeV, MeV, keV, eV, meV, ueV}

	const double HBAR	= 6.582119514e-16;	//(eV)
	const double K_B	= 8.6173324e-5;		//(eV/K)

	double hbar	= HBAR;
	double k_b	= K_B;
private:
};

};

#define
