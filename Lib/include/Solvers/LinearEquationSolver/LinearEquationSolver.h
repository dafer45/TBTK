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
 *  @file LinearEquationSolver.h
 *  @brief Solves Hx = y for x, where H is given by the Model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LINEAR_EQUATION_SOLVER
#define COM_DAFER45_TBTK_LINEAR_EQUATION_SOLVER

#include "Communicator.h"
#include "Model.h"
#include "Solver.h"

#include <complex>
#include <vector>

namespace TBTK{

class LinearEquationSolver : public Solver, public Communicator{
public:
	/** Constructor */
	LinearEquationSolver();

	/** Destructor. */
	virtual ~LinearEquationSolver();

	/** Modes. */
	enum class Mode {LU, ConjugateGradient};

	/** Set mode. */
	void setMode(Mode mode);

	/** Solve. */
	std::vector<std::complex<double>> solve(
		const std::vector<std::complex<double>> &y
	);
private:
	/** pointer to array containing Hamiltonian. */
	std::complex<double> *hamiltonian;

	/** Mode. */
	Mode mode;

	/** Solve using LU-decomposition. */
	std::vector<std::complex<double>> solveLU(
		const std::vector<std::complex<double>> &y
	);

	/** Solve using conjugate gradient. */
	std::vector<std::complex<double>> solveConjugateGradient(
		const std::vector<std::complex<double>> &y
	);
};

inline void LinearEquationSolver::setMode(Mode mode){
	this->mode = mode;
}

inline std::vector<std::complex<double>> LinearEquationSolver::solve(
	const std::vector<std::complex<double>> &y
){
	switch(mode){
	case Mode::LU:
		return solveLU(y);
	case Mode::ConjugateGradient:
		return solveConjugateGradient(y);
	default:
		TBTKExit(
			"LinearEquationSolver::solve()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace TBTK

#endif
