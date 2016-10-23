/** @file ChebyshevSolverGPUDummy.cpp
 *  @brief Dummy functions to allow for compilation without GPU support
 *
 *  @author Kristofer Björnson
 */

#include "../../include/ChebyshevSolver.h"
#include "../../include/Streams.h"

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
