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

#include "ChebyshevSolver.h"
#include "Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{

void ChebyshevSolver::calculateCoefficientsGPU(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	TBTKExit(
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevSolver::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	TBTKExit(
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevSolver::loadLookupTableGPU(){
	TBTKExit(
		"ChebyshevSolver::loadLookupTableGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevSolver::destroyLookupTableGPU(){
	TBTKExit(
		"ChebyshevSolver::destroyLookupTableGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

void ChebyshevSolver::generateGreensFunctionGPU(
	complex<double> *greensFunction,
	complex<double> *coefficients,
	ChebyshevSolver::GreensFunctionType type
){
	TBTKExit(
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"GPU Not supported.",
		"Install with GPU support or use CPU version."
	);
}

/*void ChebyshevSolver::createDeviceTableGPU(){
	numDevices = 0;
}

void ChebyshevSolver::destroyDeviceTableGPU(){
}*/

};	//End of namespace TBTK
