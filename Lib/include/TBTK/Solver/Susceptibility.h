/* Copyright 2017 Kristofer Björnson
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
 *  @file Suscesptibility.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_SUSCEPTIBILITY
#define COM_DAFER45_TBTK_SOLVER_SUSCEPTIBILITY

#include "TBTK/RPA/MomentumSpaceContext.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Solver{

class Susceptibility : public Solver{
public:
	/** List of algorithm identifiers. Officilly supported algorithms are
	 *  given unique identifiers. Algorithms not (yet) supported should
	 *  make sure they use an identifier that does not clash with the
	 *  officially supported ones. [ideally a large random looking number
	 *  (magic number) to also minimize accidental clashes with other
	 *  algorithms that are not (yet) supported. */
	enum Algorithm {
		Lindhard = 0,
		Matsubara = 1,
		RPA = 2
	};

	/** Constructor. */
	Susceptibility(
		Algorithm algorithm,
		const RPA::MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	virtual ~Susceptibility();

	/** Calculate the susceptibility. */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const Index &index,
		const std::vector<std::complex<double>> &energies
	) = 0;

	const RPA::MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the algorithm used to calculate the susceptibility. */
	Algorithm getAlgorithm() const;

	/** Set to true if the susceptibility energies can be assumed
	 *  to be inversion symmetric in the complex plane.
	 *
	 *  Important note:
	 *  Only set this to true if the energies passed to
	 *  setSusceptibilityEnergies() are on the form
	 *  (-E_n, -E_{n-1}, ..., E_{n-1}, E_n). Setting this flag to
	 *  true without fullfilling this condition will result in
	 *  undefined behavior. */
	void setEnergiesAreInversionSymmetric(
		bool energiesAreInversionSymmetric
	);

	/** Get whether the susceptibility energies are inversion
	 *  symmetric. */
	bool getEnergiesAreInversionSymmetric() const;
private:
	/** Algorithm. */
	Algorithm algorithm;

	/** Flag indicating whether the the energies in
	 *  susceptibilityEnergies are inversion symmetric in the
	 *  complex plane. */
	bool energiesAreInversionSymmetric;

	/** Momentum space context. */
	const RPA::MomentumSpaceContext *momentumSpaceContext;
};

inline const RPA::MomentumSpaceContext& Susceptibility::getMomentumSpaceContext(
) const{
	return *momentumSpaceContext;
}

inline Susceptibility::Algorithm Susceptibility::getAlgorithm() const{
	return algorithm;
}

inline void Susceptibility::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	this->energiesAreInversionSymmetric = energiesAreInversionSymmetric;
}

inline bool Susceptibility::getEnergiesAreInversionSymmetric(
) const{
	return energiesAreInversionSymmetric;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
