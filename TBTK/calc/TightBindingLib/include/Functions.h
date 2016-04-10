/** @package TBTKcalc
 *  @file Functions.h
 *  @brief Collection of physically relevant functions
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FUNCTIONS
#define COM_DAFER45_TBTK_FUNCTIONS

#include "UnitHandler.h"
#include <math.h>

namespace TBTK{

class Functions{
public:
	/** Fermi-Dirac distribution. */
	static double fermiDiracDistribution(double energy, double mu, double temperature);

	/** Bose-Einstein distribution. */
	static double boseEinsteinDistribution(double energy, double mu, double temperature);
private:
};

inline double Functions::fermiDiracDistribution(double energy, double mu, double temperature){
	double e = UnitHandler::convertEnergy(energy - mu);
	double t = UnitHandler::convertTemperature(temperature);
	return 1./(exp(e/(UnitHandler::getK_b()*t)) + 1.);
}

inline double Functions::boseEinsteinDistribution(double energy, double mu, double temperature){
	double e = UnitHandler::convertEnergy(energy - mu);
	double t = UnitHandler::convertTemperature(temperature);
	return 1./(exp(e/(UnitHandler::getK_b()*t)) - 1.);
}

};	//End of namespace TBTK

#endif
