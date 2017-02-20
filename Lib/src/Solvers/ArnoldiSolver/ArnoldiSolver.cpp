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

/** @file ArnoldiSolver.cpp
 *
 *  @author Kristofer Björnson
 */

/* Note: The ArnoldiSolver is based on the ARPACK driver zndrv2.f, which can
 * be found at EXAMPLES/COMPLEX/zndrv2.f in the ARPACK source tree. The main
 * loop closely resembles zndrv2.f.
 * ARPACK can be downloaded from http://www.caam.rice.edu/software/ARPACK/.
 * See http://www.caam.rice.edu/software/ARPACK/UG/node138.html for more
 * information about parameters. */

#include "../../../include/Solvers/ArnoldiSolver/ArnoldiSolver.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <iostream>

using namespace std;

namespace TBTK{

ArnoldiSolver::ArnoldiSolver(){
	model = NULL;

	//Arnoldi variables (ARPACK)
	calculateEigenVectors = false;
	numEigenValues = 0;
	numLanczosVectors = 0;
	shift = 0.;
	tolerance = 0.;
	maxIterations = 20;
	residuals = NULL;
	eigenValues = NULL;
	eigenVectors = NULL;

	//SuperLU variables
	hamiltonian = NULL;
	rowPermutations = NULL;
	colPermutations = NULL;
	options = NULL;
	stat = NULL;
	vector = NULL;
	lowerTriangular = NULL;
	upperTriangular = NULL;
}

ArnoldiSolver::~ArnoldiSolver(){
	//Free Arnoldi variables (ARPACK)
	if(residuals != NULL)
		delete [] residuals;
	if(eigenValues != NULL)
		delete [] eigenValues;
	if(eigenVectors != NULL)
		delete [] eigenVectors;

	//Free SuperLU variables
	if(hamiltonian != NULL)
		Destroy_CompCol_Matrix(hamiltonian);
	if(rowPermutations != NULL)
		delete [] rowPermutations;
	if(colPermutations != NULL)
		delete [] colPermutations;
	if(options != NULL)
		delete options;
	if(stat != NULL)
		delete stat;
	if(vector != NULL)
		Destroy_Dense_Matrix(vector);
	if(lowerTriangular != NULL)
		Destroy_SuperNode_Matrix(lowerTriangular);
	if(upperTriangular != NULL)
		Destroy_CompCol_Matrix(upperTriangular);
}

//ARPACK function for performing single Arnoldi iteration step
extern "C" void znaupd_(
        int                     *IDO,
        char                    *BMAT,
        int                     *N,
        char                    *WHICH,
        int                     *NEV,
        double                  *TOL,
        complex<double>         *RESID,
        int                     *NCV,
        complex<double>         *V,
        int                     *LDV,
        int                     *IPARAM,
        int                     *IPNTR,
        complex<double>         *WORKD,
        complex<double>         *WORKL,
        int                     *LWORKL,
        double                  *RWORK,
        int                     *INFO
);

//ARPACK function for extracting calculated eigenvalues and eigenvectors
extern "C" void zneupd_(
        int                     *RVEC,
        char                    *HOWMANY,
        int                     *SELECT,
        complex<double>         *D,
        complex<double>         *Z,
        int                     *LDZ,
        complex<double>         *sigma,
        complex<double>         *WORKev,
        char                    *BMAT,
        int                     *N,
        char                    *WHICH,
        int                     *NEV,
        double                  *TOL,
        complex<double>         *RESID,
        int                     *NCV,
        complex<double>         *V,
        int                     *LDV,
        int                     *IPARAM,
        int                     *IPNTR,
        complex<double>         *WORKD,
        complex<double>         *WORKL,
        int                     *LWORKL,
        double                  *RWORK,
        int                     *INFO
);

void ArnoldiSolver::run(){
	Streams::out << "Running ArnoldiSovler.\n";
	TBTKAssert(
		model != NULL,
		"ArnoldiSolver::run()",
		"No model set.",
		"Use ArnoldSolver::setModel() to set model."
	);

	init();
	arnoldiLoop();
	sort();
}

void ArnoldiSolver::arnoldiLoop(){
	TBTKAssert(
		numEigenValues > 0,
		"ArnoldiSolver::arnoldiLoop()",
		"The number of eigenvalues must be larger than 0.",
		""
	);
	TBTKAssert(
		numLanczosVectors >= numEigenValues + 2,
		"ArnoldiSolver::arnoldiLoop()",
		"The number of Lanczos vectors must be at least two larger"
		<< " than the number of eigenvalues (" << numEigenValues
		<< ").",
		""
	);

	int basisSize = model->getBasisSize();

	//I = Standard eigenvalue problem Ax = lambda*x
	char bmat[1] = {'I'};
	//Which Ritz value of operator to compute, LM = compute the
	//numEigenValues largest (in magnitude) eigenvalues.
	char which[2] = {'L', 'M'};

	complex<double> sigma = shift;

	//Reverse communication variable.
	int ido = 0;
	//info=0 indicates that a random vector is used to start the Arnoldi iteration
	int info = 0;

	//Integer parameters used by ARPACK
	int iparam[11];
	//Exact shifts with respect to the current Hessenberg matrix
	iparam[0] = 1;
	//Maximum number of Arnoldi iterations
	iparam[2] = maxIterations;
	//Use mode 3 of ZNAUPD	(shift and invert)
	iparam[6] = 3;

	//Integer "pointer" used by ARPACK to index into workd
	int ipntr[14];

	//Allocate workspaces and output
	int worklSize = 3*numLanczosVectors*numLanczosVectors + 5*numLanczosVectors;
	residuals = new complex<double>[basisSize];
	complex<double> *lanczosVectors = new complex<double>[basisSize*numLanczosVectors];
	complex<double> *workd = new complex<double>[3*basisSize];
	complex<double> *workl = new complex<double>[worklSize];
	double *rwork = new double[basisSize];
//	int *select = new int[numEigenValues];	//Need to be allocated, but not initialized as long as howMany = 'A' in call to zneupd_
	int *select = new int[numLanczosVectors];	//Need to be allocated, but not initialized as long as howMany = 'A' in call to zneupd_
	eigenValues = new complex<double>[numEigenValues];
	if(calculateEigenVectors)
		eigenVectors = new complex<double>[numEigenValues*model->getBasisSize()];
//		eigenVectors = new complex<double>[(numEigenValues+1)*model->getBasisSize()];
	complex<double> *workev = new complex<double>[2*numLanczosVectors];

	//Main loop ()
	int counter = 0;
	while(true){
		Streams::out << "." << flush;
		if(counter%10 == 9)
			Streams::out << " ";
		if(counter%50 == 49)
			Streams::out << "\n";

		TBTKAssert(
			counter++ <= maxIterations,
			"ArnoldiSolver::arnoldiLoop()",
			"Maximum number of iterations reached.",
			""
		);

		//Calculate one more Lanczos vector
		znaupd_(
			&ido,
			bmat,
			&basisSize,
			which,
			&numEigenValues,
			&tolerance,
			residuals,
			&numLanczosVectors,
			lanczosVectors,
			&basisSize,
			iparam,
			ipntr,
			workd,
			workl,
			&worklSize,
			rwork,
			&info
		);

		if(ido == -1 || ido == 1){
			//Solve x = (A - sigma*I)^{-1}b, where b = workd[ipntr[0]] and x = workd[ipntr[1]]
			for(int n = 0; n < basisSize; n++)
				((complex<double>*)((DNformat*)vector->Store)->nzval)[n] = workd[ipntr[0] + n];

			trans_t transpose = NOTRANS;
			zgstrs(
				transpose,
				lowerTriangular,
				upperTriangular,
				colPermutations,
				rowPermutations,
				vector,
				stat,
				&info
			);

			for(int n = 0; n < basisSize; n++)
				workd[ipntr[1] + n] = ((complex<double>*)((DNformat*)vector->Store)->nzval)[n];

			continue;
		}

		TBTKAssert(
			info >= 0,
			"ArnoldiSolver::arnoldiLoop()",
			"Error with _naupd, info = " << info << ".",
			"Check the documentation in _naupd."
		);

		//A = Compute numberOfEigenValues Ritz vectors
		char howMany = 'A';
		//Error message
		int ierr;
		//Convert flag from bool to int
		int calculateEigenVectorsBool = calculateEigenVectors;

		//Extract eigenvalues and eigenvectors
		zneupd_(
			&calculateEigenVectorsBool,
			&howMany,
			select,
			eigenValues,
			eigenVectors,
			&basisSize,
			&sigma,
			workev,
			bmat,
			&basisSize,
			which,
			&numEigenValues,
			&tolerance,
			residuals,
			&numLanczosVectors,
			lanczosVectors,
			&basisSize,
			iparam,
			ipntr,
			workd,
			workl,
			&worklSize,
			rwork,
			&ierr
		);

		if(ierr != 0){
			Streams::err << "\nError with _neupd, info = " << ierr << ". Check the documentation of _neupd.";
			exit(1);
		}
		else{
			double numAccurateEigenValues = iparam[4];	//With respect to tolerance
			Streams::out << "\nNumber of accurately converged eigenvalues: " << numAccurateEigenValues << "\n";
			//Calculate |Ax - lambda*x| here
			//...
		}

		break;

		if(info == 1){
			Streams::log << "Warning: Maximum number of iterations reached.\n";
			break;
		}
		else if(info == 3){
			Streams::log << "Warning: No shifts could be applied during implicit Arnoldi update. Try increasing numEigenValues.\n";
		}
	}
	Streams::out << "\n";

	//Free memory
	delete [] lanczosVectors;
	delete [] workd;
	delete [] workl;
	delete [] rwork;
	delete [] select;
	delete [] workev;
}

void ArnoldiSolver::init(){
	//Get matrix representation on COO format
	int basisSize = model->getBasisSize();
	int numMatrixElements = model->getHoppingAmplitudeSet()->getNumMatrixElements();
	const int *cooRowIndices = model->getHoppingAmplitudeSet()->getCOORowIndices();
	const int *cooColIndices = model->getHoppingAmplitudeSet()->getCOOColIndices();
	const complex<double> *cooValues = model->getHoppingAmplitudeSet()->getCOOValues();
	TBTKAssert(
		cooRowIndices != NULL && cooColIndices != NULL,
		"ArnoldiSolver::init()",
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

	//Apply shift to diagonal elements
	currentColumn = -1;
	int nextDiagonalElement = 0;
	for(int n = 0; n < numMatrixElements; n++){
		if(cooRowIndices[n] > currentColumn){
			currentColumn = cooRowIndices[n];
		}

		TBTKAssert(
			nextDiagonalElement <= currentColumn,
			"ArnoldiSolver::init()",
			"Missing diagonal hopping amplitude in the Hamiltonian.",
			"The ArnoldiSolver requires all diagonal entries to be set even if they are zero. Use Model::addHA() to add HoppingAmplitudes for all diagonal terms."
		);

		if(cooColIndices[n] == currentColumn)
			valuesH[n].r -= shift;
	}

	//Create Hamiltonian
	hamiltonian = new SuperMatrix;
	zCreate_CompCol_Matrix(
		hamiltonian,
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
	colPermutations = new int[basisSize];
	rowPermutations = new int[basisSize];

	//Initialize SuperLU
	options = new superlu_options_t();
	set_default_options(options);
	options->ColPerm = NATURAL;

	stat = new SuperLUStat_t();
	StatInit(stat);

	//Create vector
	doublecomplex *valuesV = new doublecomplex[basisSize];
	vector = new SuperMatrix();
	zCreate_Dense_Matrix(
		vector,
		basisSize,
		1,	//Number of B vectors
		valuesV,
		basisSize,
		SLU_DN,
		SLU_Z,
		SLU_GE
	);

	//Allocate lower and upper triangular matrices
	lowerTriangular = new SuperMatrix();
	upperTriangular = new SuperMatrix();

	//Perform LU factorization of the Hamiltonian
	performLUFactorization();
}

void ArnoldiSolver::performLUFactorization(){
	//LU factorization performed in accordance with the procedure used in
	//zgssv.c in SuperLU 5.2.1. See this file for further details.
	int cPermSpec = options->ColPerm;
	if(cPermSpec != MY_PERMC && options->Fact == DOFACT)
		get_perm_c(cPermSpec, hamiltonian, colPermutations);

	int *etree = new int[hamiltonian->ncol];

	//Hamiltonian post multiplied by colPermutations, i.e. H*Pc
	SuperMatrix hamiltonianCP;
	sp_preorder(
		options,
		hamiltonian,
		colPermutations,
		etree,
		&hamiltonianCP
	);

	int panelSize = sp_ienv(1);
	int relax = sp_ienv(2);

	int lwork = 0;

	GlobalLU_t Glu;
	int info;
	zgstrf(
		options,
		&hamiltonianCP,
		relax,
		panelSize,
		etree,
		NULL,
		lwork,
		colPermutations,
		rowPermutations,
		lowerTriangular,
		upperTriangular,
		&Glu,
		stat,
		&info
	);

	if(info != 0){
		if(info < 0){
			TBTKExit(
				"ArnoldiSolver:performLUFactorization()",
				"zgstrf returned with info = " << info << ".",
				"Contact developer, argument " << -info << " to zgstrf has invalid value."
			);
		}
		else{
			if(info <= hamiltonianCP.ncol){
				TBTKExit(
					"ArnoldiSolver:performLUFactorization()",
					"LU factorization is exactly signular. Element U(" << info << ", " << info << ") is zero.",
					"Try adding a small perturbation to the Hamiltonian."
				);
			}
			else{
				TBTKExit(
					"ArnoldiSolver:performLUFactorization()",
					"Memory allocation error.",
					""
				);
			}
		}
	}

	delete [] etree;
	Destroy_CompCol_Permuted(&hamiltonianCP);
}

void ArnoldiSolver::sort(){
	complex<double> *workspace = new complex<double>[numEigenValues];
	for(int n = 0; n < numEigenValues; n++)
		workspace[n] = eigenValues[n];

	int *order = new int[numEigenValues];
	int *orderWorkspace = new int[numEigenValues];
	for(int n = 0; n < numEigenValues; n++){
		order[n] = n;
		orderWorkspace[n] = n;
	}

	mergeSortSplit(
		workspace,
		eigenValues,
		orderWorkspace,
		order,
		0,
		numEigenValues
	);

	if(calculateEigenVectors){
		complex<double> *eigenVectorsWorkspace = new complex<double>[numEigenValues*model->getBasisSize()];
		for(int n = 0; n < numEigenValues*model->getBasisSize(); n++)
			eigenVectorsWorkspace[n] = eigenVectors[n];

		for(int n = 0; n < numEigenValues; n++)
			for(int c = 0; c < model->getBasisSize(); c++)
				eigenVectors[n*model->getBasisSize() + c] = eigenVectorsWorkspace[order[n]*model->getBasisSize() + c];

		delete [] eigenVectorsWorkspace;
	}

	delete [] order;
	delete [] orderWorkspace;
	delete [] workspace;
}

void ArnoldiSolver::mergeSortSplit(
	complex<double> *dataIn,
	complex<double> *dataOut,
	int *orderIn,
	int *orderOut,
	int first,
	int end
){
	if(end - first < 2)
		return;

	int middle = (end + first)/2;

	mergeSortSplit(dataOut, dataIn, orderOut, orderIn, first, middle);
	mergeSortSplit(dataOut, dataIn, orderOut, orderIn, middle, end);

	mergeSortMerge(dataIn, dataOut, orderIn, orderOut, first, middle, end);
}

void ArnoldiSolver::mergeSortMerge(
	complex<double> *dataIn,
	complex<double> *dataOut,
	int *orderIn,
	int *orderOut,
	int first,
	int middle,
	int end
){
	int i = first;
	int j = middle;
	for(int k = first; k < end; k++){
		if(i < middle && (j >= end || real(dataIn[i]) <= real(dataIn[j]))){
			dataOut[k] = dataIn[i];
			orderOut[k] = orderIn[i];
			i++;
		}
		else{
			dataOut[k] = dataIn[j];
			orderOut[k] = orderIn[j];
			j++;
		}
	}
}

};	//End of namespace TBTK
