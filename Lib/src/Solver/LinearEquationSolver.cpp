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

#include "slu_zdefs.h"

using namespace std;

namespace TBTK{
namespace Solver{

LinearEquationSolver::LinearEquationSolver() : Communicator(true){
	hamiltonian = NULL;
	mode = Mode::LU;
}

LinearEquationSolver::~LinearEquationSolver(){
	if(hamiltonian != NULL)
		delete [] hamiltonian;
}

vector<complex<double>> LinearEquationSolver::solveLU(
	const vector<complex<double>> &y
){
	TBTKAssert(
		(int)y.size() == getModel().getBasisSize(),
		"LinearEquationSolver::solve()",
		"'y' must have the same size as the basis size of the Model.",
		"'y' has size '" << y.size() << "', while the Model has basis"
		<< " size '" << getModel().getBasisSize() << "'."
	);

	//Get matrix representation of COO format
	const Model &model = getModel();
	int basisSize = model.getBasisSize();
	int numMatrixElements = model.getHoppingAmplitudeSet().getNumMatrixElements();
	const int *cooRowIndices = model.getHoppingAmplitudeSet().getCOORowIndices();
	const int *cooColIndices = model.getHoppingAmplitudeSet().getCOOColIndices();
	const complex<double> *cooValues = model.getHoppingAmplitudeSet().getCOOValues();
	TBTKAssert(
		cooRowIndices != nullptr && cooColIndices != nullptr,
		"LinearEquationSolver::solve()",
		"COO format not constructed.",
		"Use Model::constructCOO() to construct COO format."
	);

	//Copy rowIndices (Note that COO is on row major order. Therefore
	//columns and rows are interchanged and values complex conjugated.)
	int *rowIndicesH = new int[numMatrixElements];
	doublecomplex *valuesH = new doublecomplex[numMatrixElements];
	for(int n = 0; n < numMatrixElements; n++){
		rowIndicesH[n] = cooColIndices[n];
		valuesH[n].r = real(cooValues[n]);
		valuesH[n].i = -imag(cooValues[n]);
	}

	//Create column pointer for compressed format used by SuperLU (Note
	//that COO is on row major order. Therefore columns and rows are
	//interchanged and values complex conjugated.)
	int *colPointersH = new int[basisSize+1];
	int currentColumn = -1;
	for(int n = 0; n < numMatrixElements; n++){
		if(cooRowIndices[n] > currentColumn){
			currentColumn = cooRowIndices[n];
			colPointersH[currentColumn] = n;
		}
	}
	colPointersH[basisSize] = numMatrixElements;

	//Create Hamiltonian
	SuperMatrix hamiltonian;
	zCreate_CompCol_Matrix(
		&hamiltonian,
		basisSize,
		basisSize,
		numMatrixElements,
		valuesH,
		rowIndicesH,
		colPointersH,
		SLU_NC,
		SLU_Z,
		SLU_GE
	);

	//Allocate permutation matrices
	int *colPermutations = new int[basisSize];
	int *rowPermutations = new int[basisSize];

	//Initialize SuperLU
	superlu_options_t options;
	set_default_options(&options);
	options.ColPerm = NATURAL;
	SuperLUStat_t stat;
	StatInit(&stat);

	//Create vector
	doublecomplex *valuesV = new doublecomplex[basisSize];
	for(int n = 0; n < basisSize; n++){
		valuesV[n].r = real(y[n]);
		valuesV[n].i = imag(y[n]);
	}
	SuperMatrix B;
	zCreate_Dense_Matrix(
		&B,
		basisSize,
		1,	//Number of B vectors
		valuesV,
		basisSize,
		SLU_DN,
		SLU_Z,
		SLU_GE
	);

	SuperMatrix lowerTriangular;
	SuperMatrix upperTriangular;
	int info;
	zgssv(
		&options,
		&hamiltonian,
		colPermutations,
		rowPermutations,
		&lowerTriangular,
		&upperTriangular,
		&B,
		&stat,
		&info
	);

	if(info != 0){
		if(info < 0){
			TBTKExit(
				"LinearEquationSolver::solve()",
				"zzssv returned with info = " << info << ".",
				"Contact developer, argument " << -info << " to zzssv has invalid value."
			);
		}
		else{
			if(info <= hamiltonian.ncol){
				TBTKExit(
					"LinearEquationSolver::solve()",
					"LU factorization is exactly signular. Element U(" << info << ", " << info << ") is zero.",
					"Try adding a small perturbation to the Hamiltonian."
				);
			}
			else{
				TBTKExit(
					"LinearEquationSolver::solve()",
					"Memory allocation error.",
					""
				);
			}
		}
	}

	doublecomplex *answer = (doublecomplex*)((DNformat*)B.Store)->nzval;

	vector<complex<double>> result;
	for(int n = 0; n < basisSize; n++)
		result.push_back(complex<double>(answer[n].r, answer[n].i));

	return result;
}

vector<complex<double>> LinearEquationSolver::solveConjugateGradient(
	const vector<complex<double>> &y
){
	TBTKNotYetImplemented("LinearEquationSolver::solveConjugateGradient()");
}

};	//End of namespace Solver
};	//End of namespace TBTK
