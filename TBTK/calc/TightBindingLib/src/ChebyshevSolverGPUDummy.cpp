/** @file ChebyshevSolverGPUDummy.cpp
 *  @brief Dummy functions to allow for compilation without GPU support
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"
#include "../include/Streams.h"

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
	Util::Streams::err << "Error in ChebyshevSolver::calculateCoefficientsGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	Util::Streams::err << "Error in ChebyshevSolver::calculateCoefficientsGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::loadLookupTableGPU(){
	Util::Streams::err << "Error in ChebyshevSolver::loadLookupTableGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::destroyLookupTableGPU(){
	Util::Streams::err << "Error in ChebyshevSolver::destroyLookupTableGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::generateGreensFunctionGPU(
	complex<double> *greensFunction,
	complex<double> *coefficients,
	ChebyshevSolver::GreensFunctionType type
){
	Util::Streams::err << "Error in ChebyshevSolver::generateGreensFunctionGPU: GPU Not supported.\n";
	exit(1);
}

/*void ChebyshevSolver::createDeviceTableGPU(){
	numDevices = 0;
}

void ChebyshevSolver::destroyDeviceTableGPU(){
}*/

};	//End of namespace TBTK
