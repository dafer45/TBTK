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

#ifndef COM_DAFER45_TBTK_SOLVER_LINEAR_EQUATION_SOLVER
#define COM_DAFER45_TBTK_SOLVER_LINEAR_EQUATION_SOLVER

#include "TBTK/Communicator.h"
#include "TBTK/Matrix.h"
#include "TBTK/Model.h"
#include "TBTK/Solver/Solver.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Solver{

class LinearEquationSolver : public Solver, public Communicator{
public:
	/** Constructor */
	LinearEquationSolver();

	/** Destructor. */
	virtual ~LinearEquationSolver();

	/** Run calculation. */
	void run();

	/** Get amplitude for given Index. */
	const std::complex<double> getAmplitude(const Index &index) const;

	/** Get result. */
	const Matrix<std::complex<double>>& getResult() const;
private:
	/** The right hand side of the equation. */
	Matrix<std::complex<double>> source;
};

inline const std::complex<double> LinearEquationSolver::getAmplitude(
	const Index &index
) const{
	return source.at(getModel().getBasisIndex(index), 0);
}

inline const Matrix<std::complex<double>>& LinearEquationSolver::getResult() const{
	return source;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
