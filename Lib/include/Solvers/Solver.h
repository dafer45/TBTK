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

#ifndef COM_DAFER45_TBTK_SOLVER
#define COM_DAFER45_TBTK_SOLVER

#include "Model.h"

namespace TBTK{

class Solver{
public:
	/** Constructor */
	Solver();

	/** Destructor. */
	virtual ~Solver();

	/** Set model to work on. */
	virtual void setModel(Model *model);

	/** Get model. */
	Model* getModel();

	/** Get model. */
	const Model* getModel() const;
private:
	/** Model to work on. */
	Model *model;
};

inline void Solver::setModel(Model *model){
	this->model = model;
}

inline Model* Solver::getModel(){
	return model;
}

inline const Model* Solver::getModel() const{
	return model;
}

};	//End of namespace TBTK

#endif
