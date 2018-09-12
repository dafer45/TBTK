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
	 *  @param denumeratorDegree The degree of the denumerator polynomial. */
	void setDenumeratorDegree(unsigned int denumeratorDegree);

	/** Convert a Matsubara Green's function from the imaginary to the real
	 *  axis. */
	Property::GreensFunction convert(
		const Property::GreensFunction &greensFunction
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

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
