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

/** @package TBTKcalc
 *  @file Sto3g.h
 *  @brief STO-3G state.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_STO_3G
#define COM_DAFER45_TBTK_STO3G

#include "TBTK/AbstractState.h"
#include "TBTK/Atom.h"
#include "TBTK/KineticOperator.h"
#include "TBTK/NuclearPotentialOperator.h"
#include "TBTK/HartreeFockPotentialOperator.h"

namespace TBTK{

class Sto3g : public AbstractState{
public:
	/** Constructor.
	 *
	 *  @param slaterExponent The slater exponent.
	 *  @param coordinates The coordinate at which the state is centered.
	 *  @param index The state index.
	 *  @param spinIndex Flag indicating which subindex that is a spin
	 *  index. If set to -1, the state is considered spinless. */
	Sto3g(
		double slaterExponent,
		const std::vector<double> &coordinates,
		const Index &index,
		int spinIndex = -1
	);

	/** Implements AbstracState::clone(). */
	virtual Sto3g* clone() const;

	/** Implements AbstractState::getOverlap(). */
	virtual std::complex<double> getOverlap(
		const AbstractState &ket
	) const;

	/** Implements AbstractState::getMatrixElement(). */
	virtual std::complex<double> getMatrixElement(
		const AbstractState &ket,
		const AbstractOperator &o
	) const;
private:
	/** The contraction coefficients. */
	double contractionCoefficients[3] = {0.444635, 0.535328, 0.154329};

	/** The Gaussian exponents. */
	double gaussianExponents[3];

	/** Flag indicating the subindex that is a spin index. If equal to -1,
	 *  the state is considered spinless. */
	int spinIndex;

	/** Calculates the matrix element for the KineticOperator.
	 *
	 *  @param ket The ket in the matrix element.
	 *  @param o The operator to calculate the matrix element for.
	 *
	 *  @return The matrix element \f$<bra|-(hbar^2/2m)\nabla^2|ket>\f$,
	 *  where this state is the bra. */
	std::complex<double> getKineticTerm(
		const Sto3g &ket,
		const KineticOperator &o
	) const;

	/** Calculates the matrix element for the NuclearPotentialOperator.
	 *
	 *  @param ket The ket in the matrix element.
	 *  @param o The operator to calculate the matrix element for.
	 *
	 *  @return The matrix element \f$<bra|-e^2/(4\pi\epsilon_0 r)|ket>\f$,
	 *  where this state is the bra. */
	std::complex<double> getNuclearPotentialTerm(
		const Sto3g &ket,
		const NuclearPotentialOperator &o
	) const;

	/** Calculates the matrix element for the HartreeFockPotentialOperator.
	 *
	 *  @param ket The ket in the matrix element.
	 *  @param o The operator to calculate the matrix element for.
	 *
	 *  @return The matrix element \f$<bra|v_{HF}|ket>\f$, where this state
	 *  is the bra. */
	std::complex<double> getHartreeFockPotentialTerm(
		const Sto3g &ket,
		const HartreeFockPotentialOperator &o
	) const;

	/** Calculates a single term for the Hartree-Fock potential. The
	 *  calling function (getHartreeFockPotentialTerm()) is responsible for
	 *  performing the permuation of the state, while this function
	 *  calculates the value for the given permutation. The function also
	 *  performs the calculation in atomic units and it is the calling
	 *  functions responsibility to multiply by the appropriate prefactors.
	 *
	 *  Note: The order of the states is such that for the non-permuted
	 *  case, the states come in the order
	 *  \Psi_0^{*}(x_1)\Psi_1^{*}(x_2)\Psi_2(x_2)\Psi_3(x_1) in the
	 *  integral expression, with Psi_0 corresponding to 'this' state.
	 *
	 *  @param state1 The second state in the Hartree-Fock integral.
	 *  @param state2 The third state in the Hartree-Fock integral.
	 *  @param state3 The fourth state in the Hartree-Fock integral. */
	std::complex<double> getSingleHartreeFockTerm(
		const Sto3g &state1,
		const Sto3g &state2,
		const Sto3g &state3
	) const;

	/** Helper function for calculating matrix elements. See Appendix A in
	 *  the book Modern Quantum Chemistry: Introduction to Advanced
	 *  Electronic Structure Theory by A Szabo and NS Ostlund.
	 *
	 *  @param x Function argument.
	 *
	 *  @return The value of the function at x. */
	static double F_0(double x);
};

inline Sto3g* Sto3g::clone() const{
	return new Sto3g(*this);
}

};	//End of namespace TBTK

#endif
