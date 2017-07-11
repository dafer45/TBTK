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
 *  @file Functions.h
 *  @brief Collection of physically relevant functions
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FUNCTIONS
#define COM_DAFER45_TBTK_FUNCTIONS

#include "UnitHandler.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <math.h>

namespace TBTK{

class Functions{
public:
	/** Fermi-Dirac distribution. */
	static double fermiDiracDistribution(
		double energy,
		double mu,
		double temperature
	);

	/** Bose-Einstein distribution. */
	static double boseEinsteinDistribution(
		double energy,
		double mu,
		double temperature
	);
private:
};

inline double Functions::fermiDiracDistribution(
	double energy,
	double mu,
	double temperature
){
	double e = UnitHandler::convertEnergyNtB(energy - mu);
	double t = UnitHandler::convertTemperatureNtB(temperature);
	if(t != 0.){
		return 1./(exp(e/(UnitHandler::getK_BB()*t)) + 1.);
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

inline double Functions::boseEinsteinDistribution(
	double energy,
	double mu,
	double temperature
){
	double e = UnitHandler::convertEnergyNtB(energy - mu);
	double t = UnitHandler::convertTemperatureNtB(temperature);

	if(t != 0.){
		return 1./(exp(e/(UnitHandler::getK_BB()*t)) - 1.);
	}
	else{
		TBTKExit(
			"Functions::boseEinsteinDistribution()",
			"Bose-Einstein distribution not well behaved at T=0.",
			"Use Model::setTemperature() to set a non-zero temperature."
		);
	}
}

};	//End of namespace TBTK

#endif
