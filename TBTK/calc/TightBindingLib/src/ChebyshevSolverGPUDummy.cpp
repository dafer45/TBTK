/** @file ChebyshevSolverGPUDummy.cpp
 *  @brief Dummy functions to allow for compilation without GPU support
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"

using namespace std;

namespace TBTK{

void ChebyshevSolver::calculateCoefficientsGPU(Index to, Index from, complex<double> *coefficients, int numCoefficients, double broadening){
	cout << "Error in ChebyshevSolver::calculateCoefficientsGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::calculateCoefficientsGPU(vector<Index> &to, Index from, complex<double> *coefficients, int numCoefficients, double broadening){
	cout << "Error in ChebyshevSolver::calculateCoefficientsGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::loadLookupTableGPU(){
	cout << "Error in ChebyshevSolver::loadLookupTableGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::destroyLookupTableGPU(){
	cout << "Error in ChebyshevSolver::destroyLookupTableGPU: GPU Not supported.\n";
	exit(1);
}

void ChebyshevSolver::generateGreensFunctionGPU(complex<double> *greensFunction, complex<double> *coefficients){
	cout << "Error in ChebyshevSolver::generateGreensFunctionGPU: GPU Not supported.\n";
	exit(1);
}

};	//End of namespace TBTK
