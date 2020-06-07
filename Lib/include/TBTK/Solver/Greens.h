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
#include "TBTK/Property/SpectralFunction.h"
#include "TBTK/Property/TransmissionRate.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Calculates properties from a Green's function. */
class Greens : public Solver, public Communicator{
	TBTK_DYNAMIC_TYPE_INFORMATION(Greens)
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
	 *  @return \f$G = (G_0^{-1} + \Sigma)^{-1}\f$ */
	Property::GreensFunction calculateInteractingGreensFunction(
		const Property::SelfEnergy &selfEnergy
	) const;

	/** Calculate the spectral function.
	 *
	 *  @return \f$A = i\left(G - G^{\dagger}\right)\f$. */
	Property::SpectralFunction calculateSpectralFunction() const;

	/** Calculate the transmission.
	 *
	 *  @param selfEnergy0 The selfEnergy for the first lead.
	 *  @param selfEnergy1 The selfEnergy for the second lead.
	 *
	 *  @return The transmission from lead one to lead two. */
	Property::TransmissionRate calculateTransmissionRate(
		const Property::SelfEnergy &selfEnergy0,
		const Property::SelfEnergy &selfEnergy1
	) const;
private:
	/** Green's function to use in calculations. */
	const Property::GreensFunction *greensFunction;

	/** Helper class that can contain information about the block
	 *  structure. */
	class BlockStructure{
	public:
		bool isBlockRestricted;
		bool globalBlockIsContained;
		IndexTree containedBlocks;
	};

	/** Extract information about the Green's functions block structure. */
	BlockStructure getBlockStructure() const;

	/** Verify that the block structure contains all index pairs for those
	 *  blocks that share at least one index pair. */
	void verifyBlockStructure(const BlockStructure &blockStructure) const;

	/** Verify that the Green's function contains all Index pairs in the
	 *  block formed by the given Indices. */
	void verifyGreensFunctionContainsAllIndicesInBlock(
		const IndexTree &intraBlockIndices
	) const;

	/** Create a new Green's function with the same structure as the
	 *  Green's function that is used as input. */
	Property::GreensFunction createNewGreensFunction() const;

	/** Convert Green's function to matrix for the given energy index and
	 *  Indices. */
	void convertGreensFunctionToMatrix(
		Matrix<std::complex<double>> &matrix,
		unsigned int energy,
		const IndexTree &intraBlockIndices
	) const;

	/** Add the self-energy to the Greens'f unction matrix for the given
	 *  energy index and Indices. */
	void addSelfEnergyToGreensFunctionMatrix(
		Matrix<std::complex<double>> &matrix,
		const Property::SelfEnergy &selfEnergy,
		unsigned int energy,
		const IndexTree &intraBlockIndices
	) const;

	/** Write a Green's function matrix back to an interacting Green's
	 *  function for the given energy index and Indices. */
	void writeGreensFunctionMatrixToInteractingGreensFunction(
		Property::GreensFunction &interactingGreensFunction,
		const Matrix<std::complex<double>> &matrix,
		unsigned int energy,
		const IndexTree &intraBlockIndices
	) const;

	/** Calculate a single block of the interacting Green's function. */
	void calculateInteractingGreensFunctionSingleBlock(
		Property::GreensFunction &interactingGreensFunction,
		const Property::SelfEnergy &selfEnergy,
		const IndexTree &intraBlockIndices
	) const;
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
