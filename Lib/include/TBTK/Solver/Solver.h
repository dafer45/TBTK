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
 *  @file Solver.h
 *  @brief Base class for Solvers.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_SOLVER
#define COM_DAFER45_TBTK_SOLVER_SOLVER

#include "TBTK/Model.h"

namespace TBTK{
namespace Solver{

/** @brief Base class for Solvers.
 *
 *  The Solver is a base class for other @link Solver Solvers@endlink. See
 *  therefore the documentation for the Diagonalizer, BlockDiagonalizer,
 *  ArnoldiIterator, and ChebyshevExpander examples of specific production
 *  ready @link Solver Solvers@endlink.
 *
 *  # Example
 *  \snippet Solver/Solver.cpp Solver
 *  ## Output
 *  \snippet output/Solver/Solver.output Solver */
class Solver{
public:
	/** Constructs a Solver::Solver. */
	Solver();

	/** Destructor. */
	virtual ~Solver();

	/** Set model to solve.
	 *
	 *  @param model The Model that is to be solved. */
	virtual void setModel(Model &model);

	/** Get model.
	 *
	 *  @return The Model that the Solver is solving. */
	Model& getModel();

	/** Get model.
	 *
	 *  @return The Model that the Solver is solving. */
	const Model& getModel() const;
private:
	/** Model to work on. */
	Model *model;
};

inline void Solver::setModel(Model &model){
	this->model = &model;
}

inline Model& Solver::getModel(){
	TBTKAssert(
		model != nullptr,
		"Solver::getModel()",
		"Model not set.",
		"Use Solver::setSolver() to set model."
	);
	return *model;
}

inline const Model& Solver::getModel() const{
	TBTKAssert(
		model != nullptr,
		"Solver::getModel()",
		"Model not set.",
		"Use Solver::setSolver() to set model."
	);
	return *model;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
