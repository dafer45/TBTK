/* Copyright 2016 Kristofer Björnson
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

/** @file DiagonalizerSolverGPUDummy.cpp
 *  @brief Dummy functions to allow for compilation without GPU support
 *
 *  @author Kristofer Björnson, Andreas Theiler
 */

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Solver{

vector<complex<double>> Diagonalizer::solveGPU(
	Index to,
	Index from
){
	TBTKExit(
		"Diagonalizer::solveGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

};	//End of namespace Solver
};	//End of namespace TBTK
