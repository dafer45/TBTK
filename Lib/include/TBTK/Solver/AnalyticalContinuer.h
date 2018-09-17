/* Copyright 2018 Kristofer Björnson
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
 *  @file Greens.h
 *  @brief Calculates properties from a Green's function.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_ANALYTICAL_CONTINUER
#define COM_DAFER45_TBTK_SOLVER_ANALYTICAL_CONTINUER

#include "TBTK/Communicator.h"
#include "TBTK/Model.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Performs analytical continuation of EnergyResolved properties. */
class AnalyticalContinuer : public Solver, public Communicator{
public:
	/** Constructs a Solver::AnalyticalContinuer. */
	AnalyticalContinuer();

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

	/** Set the degree of the numerator in the Padé approximation.
	 *
	 *  @param numeratorDegree The degree of the numerator polynomial. */
	void setNumeratorDegree(unsigned int numeratorDegree);

	/** Set the degree of the denumerator in the Padé approximation.
	 *
	 *  @param denumeratorDegree The degree of the denumerator polynomial.
	 */
	void setDenumeratorDegree(unsigned int denumeratorDegree);

	/** Set the size of the energy infinitesimal that is used to deform the
	 *  contour in for example the Retarded and advanced Green's functions.
	 *
	 *  @param energyInfinitesimal The energy infinitesimal. */
	void setEnergyInfinitesimal(double energyInfinitesimal);

	/** Set the energy shift to be used in the polynomial expansion.
	 *
	 *  @param energyShift Energy shift to be applied in the polynomial
	 *  expansion. */
	void setEnergyShift(std::complex<double> energyShift);

	/** Set the scale factor to be used in the polynomial expansion.
	 *
	 *  @param scaleFactor Scale factor to be applied in the polynomial
	 *  expansion. */
	void setScaleFactor(double scaleFactor);

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

	/** The degree of the numerator polynomial in the Padé approximation.
	 */
	unsigned int numeratorDegree;

	/** The degree of the denumerator polynomial in the Padé approximation.
	 */
	unsigned int denumeratorDegree;

	/** Default energy infinitesimal. */
	static constexpr double ENERGY_INFINITESIMAL = 0;

	/** The energy infinitesimal \f$\delta\f$ that is used to deform the
	 *  contour retarded and advanced Green's functions, etc. */
	double energyInfinitesimal;

	/** Energy shift to use in the polynomial expansion. */
	std::complex<double> energyShift;

	/** Scale factor to use in the polynomial expansion. */
	double scaleFactor;

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

inline void AnalyticalContinuer::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int resolution
){
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

inline void AnalyticalContinuer::setNumeratorDegree(
	unsigned int numeratorDegree
){
	this->numeratorDegree = numeratorDegree;
}

inline void AnalyticalContinuer::setDenumeratorDegree(
	unsigned int denumeratorDegree
){
	this->denumeratorDegree = denumeratorDegree;
}

inline void AnalyticalContinuer::setEnergyInfinitesimal(
	double energyInfinitesimal
){
	this->energyInfinitesimal = energyInfinitesimal;
}

inline void AnalyticalContinuer::setEnergyShift(
	std::complex<double> energyShift
){
	this->energyShift = energyShift;
}

inline void AnalyticalContinuer::setScaleFactor(double scaleFactor){
	this->scaleFactor = scaleFactor;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
