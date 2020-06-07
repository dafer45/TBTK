/* Copyright 2019 Kristofer Björnson
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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file AnalyticalContinuerContinuousFractions.h
 *  @brief Perfroms analytical continuation of EnergyResolved properties using
 *  continuous fractions.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_ANALYTICAL_CONTINUER_CONTINUOUS_FRACTIONS
#define COM_DAFER45_TBTK_SOLVER_ANALYTICAL_CONTINUER_CONTINUOUS_FRACTIONS

#include "TBTK/Communicator.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Performs analytical continuation of EnergyResolved properties using
 *  continuous fracitons. */
class AnalyticalContinuerContinuousFractions : public Solver, public Communicator{
	TBTK_DYNAMIC_TYPE_INFORMATION(AnalyticalContinuerContinuousFractions)
public:
	/** Constructs a Solver::AnalyticalContinuerContinuousFractions. */
	AnalyticalContinuerContinuousFractions();

	/** Set the energy window that the functions are continued to.
	 *
	 *  @param lowerBound The lower bound for the energy window.
	 *  @param upperBound The upper bound for the energy window.
	 *  @param resolution The number of points used to resolve the energy
	 *  window. */
	void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Set the size of the energy infinitesimal that is used to deform the
	 *  contour in for example the Retarded and advanced Green's functions.
	 *
	 *  @param energyInfinitesimal The energy infinitesimal. */
	void setEnergyInfinitesimal(double energyInfinitesimal);

	/** Convert a Matsubara Green's function from the imaginary to the real
	 *  axis.
	 *
	 *  @param greensFunction The Green's function to convert from.
	 *  @param type The Green's function type to convert to. */
	Property::GreensFunction convert(
		const Property::GreensFunction &greensFunction,
		Property::GreensFunction::Type type
	) const;
private:
	/** The lower bound for the energy window. */
	double lowerBound;

	/** The upper bound for the energy window. */
	double upperBound;

	/** The energy resolution for the energy window. */
	int resolution;

	/** Default energy infinitesimal. */
	static constexpr double ENERGY_INFINITESIMAL = 0;

	/** The energy infinitesimal \f$\delta\f$ that is used to deform the
	 *  contour retarded and advanced Green's functions, etc. */
	double energyInfinitesimal;

	/** Returns an imaginary infinitesimal that deforms the contour
	 *  according to the given Green's function type.
	 *
	 *  @param energy The real energy at which the contour should be
	 *  deformed.
	 *
	 *  @param type The type of the Green's function to obtain the contour
	 *  for.
	 *
	 *  @return The imaginary deformation of the contour at the given
	 *  energy and Green's function type. */
	std::complex<double> getContourDeformation(
		double energy,
		Property::GreensFunction::Type type
	) const;
};

inline void AnalyticalContinuerContinuousFractions::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int resolution
){
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

inline void AnalyticalContinuerContinuousFractions::setEnergyInfinitesimal(
	double energyInfinitesimal
){
	this->energyInfinitesimal = energyInfinitesimal;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
