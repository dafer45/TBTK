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
#include <iostream>

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
	double e = UnitHandler::convertEnergyNtB(energy - mu);
	double t = UnitHandler::convertTemperatureNtB(temperature);
	if(t != 0.){
		return 1./(exp(e/(UnitHandler::getK_b()*t)) + 1.);
	}
	else{
		if(e < 0.)
			return 1.;
		else if(e == 0.)
			return 0.5;
		else
			return 0.;
	}
}

inline double Functions::boseEinsteinDistribution(double energy, double mu, double temperature){
	double e = UnitHandler::convertEnergyNtB(energy - mu);
	double t = UnitHandler::convertTemperatureNtB(temperature);

	if(t != 0.){
		return 1./(exp(e/(UnitHandler::getK_b()*t)) - 1.);
	}
	else{
		std::cout << "Error in Functions::boseEinsteinDistribution(): "
				<< "Bose-Einstein distribution not well behaved at T=0. Please use\n";
		exit(1);
	}
}

};	//End of namespace TBTK

#endif
