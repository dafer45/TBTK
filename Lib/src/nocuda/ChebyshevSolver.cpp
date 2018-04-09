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

/** @file ChebyshevSolverGPUDummy.cpp
 *  @brief Dummy functions to allow for compilation without GPU support
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Solver{

vector<complex<double>> ChebyshevExpander::calculateCoefficientsGPU(
	Index to,
	Index from
){
	TBTKExit(
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

vector<vector<complex<double>>> ChebyshevExpander::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from
){
	TBTKExit(
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevExpander::loadLookupTableGPU(){
	TBTKExit(
		"ChebyshevExpander::loadLookupTableGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevExpander::destroyLookupTableGPU(){
	TBTKExit(
		"ChebyshevExpander::destroyLookupTableGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

vector<complex<double>> ChebyshevExpander::generateGreensFunctionGPU(
	const vector<complex<double>> &coefficients,
	Type type
){
	TBTKExit(
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

/*void ChebyshevExpander::createDeviceTableGPU(){
	numDevices = 0;
}

void ChebyshevExpander::destroyDeviceTableGPU(){
}*/

};	//End of namespace Solver
};	//End of namespace TBTK
