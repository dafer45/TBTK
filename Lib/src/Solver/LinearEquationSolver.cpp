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

/** @file LinearEquationSolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/LinearEquationSolver.h"
#include "TBTK/Solver/LUSolver.h"

using namespace std;

namespace TBTK{
namespace Solver{

LinearEquationSolver::LinearEquationSolver() : Communicator(true), source(0, 0){
}

LinearEquationSolver::~LinearEquationSolver(){
}

void LinearEquationSolver::run(){
	Model &model = getModel();
	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= model.getHoppingAmplitudeSet();
	const SourceAmplitudeSet &sourceAmplitudeSet
		= model.getSourceAmplitudeSet();

	LUSolver luSolver;
	luSolver.setMatrix(hoppingAmplitudeSet.getSparseMatrix());

	source = Matrix<complex<double>>(model.getBasisSize(), 1);
	for(int n = 0; n < model.getBasisSize(); n++)
		source.at(n, 0) = 0;
	for(
		SourceAmplitudeSet::ConstIterator iterator
			= sourceAmplitudeSet.cbegin();
		iterator != sourceAmplitudeSet.cend();
		++iterator
	){
		source.at(model.getBasisIndex((*iterator).getIndex()), 0)
			+= (*iterator).getAmplitude();
	}
	luSolver.solve(source);
}

};	//End of namespace Solver
};	//End of namespace TBTK
