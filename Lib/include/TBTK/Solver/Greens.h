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

#ifndef COM_DAFER45_TBTK_SOLVER_GREENS
#define COM_DAFER45_TBTK_SOLVER_GREENS

#include "TBTK/Communicator.h"
#include "TBTK/Model.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Calculates properties from a Green's function. */
class Greens : public Solver, public Communicator{
public:
	/** Constructs a Solver::Greens. */
	Greens();

	/** Destructor. */
	virtual ~Greens();

	/** Set Green's function to use for calculations.
	 *
	 *  @param greensFunction The Green's function that will be used in
	 *  calculations. */
	void setGreensFunction(const Property::GreensFunction &greensFunction);

	/** Get the Green's function.
	 *
	 *  @return The Green's function that the solver is using. */
	const Property::GreensFunction& getGreensFunction() const;

	/** Calculate a new Green's function by adding a self-energy.
	 *
	 *  @param greensFunction0 The Green's function without the
	 *  self-energy (\f$G_0\f$).
	 *  @param selfEnergy The self-energy \f$\Sigma\f$ to add to the
	 *  original Green's function.
	 *
	 *  @return \f$G = (G_0^{-1} + \Sigma)^{-1}\f$*/
	Property::GreensFunction calculateInteractingGreensFunction(
		const Property::SelfEnergy &selfEnergy
	) const;

	/** Calculate the transmission.
	 *
	 *  @param selfEnergy0 The selfEnergy for the first lead.
	 *  @param selfEnergy1 The selfEnergy for the second lead.
	 *
	 *  @return The transmission from lead one to lead two. */
	std::vector<std::complex<double>> calculateTransmission(
		const Property::SelfEnergy &selfEnergy0,
		const Property::SelfEnergy &selfEnergy1
	) const;
private:
	/** Green's function to use in calculations. */
	const Property::GreensFunction *greensFunction;
};

inline void Greens::setGreensFunction(
	const Property::GreensFunction &greensFunction
){
	this->greensFunction = &greensFunction;
}

inline const Property::GreensFunction& Greens::getGreensFunction() const{
	return *greensFunction;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
