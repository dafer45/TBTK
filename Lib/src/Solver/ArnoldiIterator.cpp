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

/** @file ArnoldiIterator.cpp
 *
 *  @author Kristofer Björnson
 */

/* Note: The ArnoldiIterator is based on the ARPACK driver zndrv2.f, which can
 * be found at EXAMPLES/COMPLEX/zndrv2.f in the ARPACK source tree. The main
 * loop closely resembles zndrv2.f.
 * ARPACK can be downloaded from http://www.caam.rice.edu/software/ARPACK/.
 * See http://www.caam.rice.edu/software/ARPACK/UG/node138.html for more
 * information about parameters. */

#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Solver{

ArnoldiIterator::ArnoldiIterator(
) :
	Communicator(false),
	matrix(SparseMatrix<complex<double>>::StorageFormat::CSC)
{
	mode = Mode::Normal;

	//Arnoldi variables (ARPACK)
	calculateEigenVectors = false;
	numEigenValues = 0;
	numLanczosVectors = 0;
	shift = 0.;
	tolerance = 0.;
	maxIterations = 20;
}

//ARPACK function for performing single Arnoldi iteration step (double)
extern "C" void dnaupd_(
	int			*IDO,
	char			*BMAT,
	int			*N,
	char			*WHICH,
	int			*NEV,
	double			*TOL,
	double			*RESID,
	int			*NCV,
	double			*V,
	int			*LDV,
	int			*IPARAM,
	int			*IPNTR,
	double			*WORKD,
	double			*WORKL,
	int			*LWORKL,
	int			*INFO
);

//ARPACK function for performing single Arnoldi iteration step (complex)
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
//(double)
extern "C" void dneupd_(
	int			*RVEC,
	char			*HOWMANY,
	int			*SELECT,
	double			*DR,
	double			*DI,
	double			*Z,
	int			*LDZ,
	double			*SIGMAR,
	double			*SIGMAI,
	double			*WORKEV,
	char			*BMAT,
	int			*N,
	char			*WHICH,
	int			*NEV,
	double			*TOL,
	double			*RESID,
	int			*NCV,
	double			*V,
	int			*LDV,
	int			*IPARAM,
	int			*IPNTR,
	double			*WORKD,
	double			*WORKL,
	int			*LWORKL,
	int			*INFO
);

//ARPACK function for extracting calculated eigenvalues and eigenvectors
//(complex)
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

void ArnoldiIterator::run(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Running ArnoldiIterator.\n";

	switch(mode){
	case Mode::Normal:
		initNormal();
		arnoldiLoop();
		break;
	case Mode::ShiftAndInvert:
		initShiftAndInvert();
		arnoldiLoop();
		break;
	default:
		TBTKExit(
			"ArnoldiIterator::run()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
	sort();
}

void ArnoldiIterator::arnoldiLoop(){
	TBTKAssert(
		numEigenValues > 0,
		"ArnoldiIterator::arnoldiLoop()",
		"The number of eigenvalues must be larger than 0.",
		""
	);
	TBTKAssert(
		numLanczosVectors >= numEigenValues + 2,
		"ArnoldiIterator::arnoldiLoop()",
		"The number of Lanczos vectors must be at least two larger"
		<< " than the number of eigenvalues (" << numEigenValues
		<< ").",
		""
	);
	TBTKAssert(
		numLanczosVectors <= getModel().getBasisSize(),
		"ArnoldiIterator::arnoldiLoop()",
		"The number of Lanczos vectors '" << numLanczosVectors << "'"
		<< " must be smaller than the basis size '"
		<< getModel().getBasisSize() << "'.",
		""
	);

	const Model &model = getModel();
	int basisSize = model.getBasisSize();

	//I = Standard eigenvalue problem Ax = lambda*x
	char bmat[1] = {'I'};
	//Which Ritz value of operator to compute, LM = compute the
	//numEigenValues largest (in magnitude) eigenvalues.
	char which[2] = {'L', 'M'};

	complex<double> sigma = shift;

	//Reverse communication variable.
	int ido = 0;
	//info=0 indicates that a random vector is used to start the Arnoldi
	//iteration.
	int info = 0;

	//Integer parameters used by ARPACK
	int iparam[11];
	//Exact shifts with respect to the current Hessenberg matrix
	iparam[0] = 1;
	//Maximum number of Arnoldi iterations
	iparam[2] = maxIterations;
	switch(mode){
	case Mode::Normal:
		//Use mode 1 of _NAUPD	(normal matrix multiplication)
		iparam[6] = 1;
		break;
	case Mode::ShiftAndInvert:
		//Use mode 3 of _NAUPD	(shift and invert)
		iparam[6] = 3;
		break;
	default:
		TBTKExit(
			"ArnoldiIterator::arnoldiLoop()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}

	//Integer "pointer" used by ARPACK to index into workd
	int ipntr[14];

	if(
		false	//Se comment below
		&& mode == Mode::ShiftAndInvert
		&& luSolver.getMatrixDataType() == LUSolver::DataType::Double
	){
		//This never happens!
		//
		//The implementation seems almost correct, but eigenvectors
		//seems to come out somewhat off. This could be due to
		//numerical instabilities, but needs further investigation
		//before this code is ready to run.

		//Allocate workspaces and output
		int worklSize = 3*numLanczosVectors*numLanczosVectors + 6*numLanczosVectors;
		residuals = CArray<complex<double>>(basisSize);	//Not used during ARPACK call
		CArray<double> residualsArpack(basisSize);
		CArray<double> lanczosVectors(basisSize*numLanczosVectors);
		CArray<double> workd(3*basisSize);
		CArray<double> workl(worklSize);
		CArray<int> select(numLanczosVectors);	//Need to be allocated, but not initialized as long as howMany = 'A' in call to dneupd_
		eigenValues = CArray<complex<double>>(numEigenValues+1);
		if(calculateEigenVectors)
			eigenVectors = CArray<complex<double>>(numEigenValues*model.getBasisSize());
		CArray<double> workev(3*numLanczosVectors);

		//Only used in Mode::ShiftAndInvert.
		Matrix<double> b(basisSize, 1);

		//Main loop ()
		int counter = 0;
		while(true){
			if(getGlobalVerbose() && getVerbose()){
				Streams::out << "." << flush;
				if(counter%10 == 9)
					Streams::out << " ";
				if(counter%50 == 49)
					Streams::out << "\n";
			}

			TBTKAssert(
				counter++ <= maxIterations,
				"ArnoldiIterator::arnoldiLoop()",
				"Maximum number of iterations reached.",
				""
			);

			//Calculate one more Lanczos vector
			dnaupd_(
				&ido,
				bmat,
				&basisSize,
				which,
				&numEigenValues,
				&tolerance,
				residualsArpack.getData(),
				&numLanczosVectors,
				lanczosVectors.getData(),
				&basisSize,
				iparam,
				ipntr,
				workd.getData(),
				workl.getData(),
				&worklSize,
				&info
			);

			checkZnaupdInfo(info);
			if(
				executeReverseCommunicationMessage(
					ido,
					basisSize,
					workd.getData(),
					ipntr,
					b
				)
			){
				break;
			}
		}
		if(getGlobalVerbose() && getVerbose())
			Streams::out << "\n";

		//A = Compute numberOfEigenValues Ritz vectors
		char howMany = 'A';
		//Error message (Set to the same value as info
		int ierr = info;
		//Convert flag from bool to int
		int calculateEigenVectorsBool = calculateEigenVectors;

		//Real and imaginary parts
		CArray<double> eigenValuesReal(numEigenValues+1);
		CArray<double> eigenValuesImag(numEigenValues+1);
		double sigmaReal = real(sigma);
		double sigmaImag = imag(sigma);

		//Ritz vectors
		CArray<double> ritzVectors(basisSize*(numEigenValues+1));

		//Extract eigenvalues and eigenvectors
		dneupd_(
			&calculateEigenVectorsBool,
			&howMany,
			select.getData(),
			eigenValuesReal.getData(),
			eigenValuesImag.getData(),
			ritzVectors.getData(),
			&basisSize,
			&sigmaReal,
			&sigmaImag,
			workev.getData(),
			bmat,
			&basisSize,
			which,
			&numEigenValues,
			&tolerance,
			residualsArpack.getData(),
			&numLanczosVectors,
			lanczosVectors.getData(),
			&basisSize,
			iparam,
			ipntr,
			workd.getData(),
			workl.getData(),
			&worklSize,
			&ierr
		);
		checkZneupdIerr(ierr);

		for(int n = 0; n < numEigenValues+1; n++){
			eigenValues[n] = complex<double>(
				eigenValuesReal[n],
				eigenValuesImag[n]
			);

			//Sigma not applied by dneupd_ in normal mode.
			//Therefore add it.
			if(mode == Mode::Normal)
				eigenValues[n] += sigma;
		}

		for(int n = 0; n < numEigenValues; n++){
			TBTKAssert(
				imag(eigenValues[n]) == 0,
				"ArnoldiIterator::arnoldiLoop()",
				"Complex eigenvalue obtained, but support not"
				<< " implemented.",
				""
			);
			for(int c = 0; c < basisSize; c++){
				eigenVectors[basisSize*n + c]
					= complex<double>(
						ritzVectors[basisSize*n + c],
						0
					);
			}
		}

		//Calculate |Ax - lambda*x| here
		//...

		for(int n = 0; n < basisSize; n++)
			residuals[n] = residualsArpack[n];
	}
	else{
		//Allocate workspaces and output
		int worklSize = 3*numLanczosVectors*numLanczosVectors + 5*numLanczosVectors;
		residuals = CArray<complex<double>>(basisSize);
		CArray<complex<double>> lanczosVectors(basisSize*numLanczosVectors);
		CArray<complex<double>> workd(3*basisSize);
		CArray<complex<double>> workl(worklSize);
		CArray<double> rwork(basisSize);
		CArray<int> select(numLanczosVectors);	//Need to be allocated, but not initialized as long as howMany = 'A' in call to zneupd_
		eigenValues = CArray<complex<double>>(numEigenValues+1);
		if(calculateEigenVectors)
			eigenVectors = CArray<complex<double>>(numEigenValues*model.getBasisSize());
		CArray<complex<double>> workev(2*numLanczosVectors);

		//Only used in Mode::ShiftAndInvert.
		Matrix<complex<double>> b(basisSize, 1);

		//Main loop ()
		int counter = 0;
		while(true){
			if(getGlobalVerbose() && getVerbose()){
				Streams::out << "." << flush;
				if(counter%10 == 9)
					Streams::out << " ";
				if(counter%50 == 49)
					Streams::out << "\n";
			}

			TBTKAssert(
				counter++ <= maxIterations,
				"ArnoldiIterator::arnoldiLoop()",
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
				residuals.getData(),
				&numLanczosVectors,
				lanczosVectors.getData(),
				&basisSize,
				iparam,
				ipntr,
				workd.getData(),
				workl.getData(),
				&worklSize,
				rwork.getData(),
				&info
			);

			checkZnaupdInfo(info);
			if(
				executeReverseCommunicationMessage(
					ido,
					basisSize,
					workd.getData(),
					ipntr,
					b
				)
			){
				break;
			}
		}
		if(getGlobalVerbose() && getVerbose())
			Streams::out << "\n";

		//A = Compute numberOfEigenValues Ritz vectors
		char howMany = 'A';
		//Error message (Set to the same value as info
		int ierr = info;
		//Convert flag from bool to int
		int calculateEigenVectorsBool = calculateEigenVectors;

		//Extract eigenvalues and eigenvectors
		zneupd_(
			&calculateEigenVectorsBool,
			&howMany,
			select.getData(),
			eigenValues.getData(),
			eigenVectors.getData(),
			&basisSize,
			&sigma,
			workev.getData(),
			bmat,
			&basisSize,
			which,
			&numEigenValues,
			&tolerance,
			residuals.getData(),
			&numLanczosVectors,
			lanczosVectors.getData(),
			&basisSize,
			iparam,
			ipntr,
			workd.getData(),
			workl.getData(),
			&worklSize,
			rwork.getData(),
			&ierr
		);
		checkZneupdIerr(ierr);

		//Sigma not applied by zneupd_ in normal mode. Therefore add
		//it.
		if(mode == Mode::Normal)
			for(int n = 0; n < numEigenValues+1; n++)
				eigenValues[n] += sigma;
		//Calculate |Ax - lambda*x| here
		//...
	}

	double numAccurateEigenValues = iparam[4]; //With respect to tolerance
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\nNumber of accurately converged eigenvalues: "
			<< numAccurateEigenValues << "\n";
	}
}

void ArnoldiIterator::checkZnaupdInfo(int info) const{
	if(info != 0){
		if(info == 1){
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"Maximum number of iterations"
				<< " reached.",
				""
			);
		}
		else if(info == 3){
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"No shifts could be applied during"
				<< " implicit Arnoldi update.",
				"Try increasing the number of Lanczos"
				<< " vectors."
			);
		}
		else if(info == -9999){
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"Unable to build an Arnoldi"
				<< " factorization.",
				"Likely input error to znaupd, which"
				<< " should never happen, contact the"
				<< " developer."
			);
		}
		else if(info < 0){
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"Input parameter '" << -info << "' to"
				<< " znaupd is invalid.",
				"This should never happen, contact the"
				<< " developer."
			);
		}
		else{
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"znaupd() exited with unknown error"
				<< " info = " << info << "'.",
				"This should never happen, contact the"
				<< " developer."
			);
		}
	}
}

bool ArnoldiIterator::executeReverseCommunicationMessage(
	int ido,
	int basisSize,
	double *workd,
	int *ipntr,
	Matrix<double> &b
){
	if(ido == -1 || ido == 1){
		switch(mode){
		case Mode::Normal:
		{
			TBTKExit(
				"ArnoldiIterator::executeReverseCommunicationMessage()",
				"Mode::Normal not implemented yet.",
				""
			);
		}
		case Mode::ShiftAndInvert:
			//Solve x = (A - sigma*I)^{-1}b, where b =
			//workd[ipntr[0]] and x = workd[ipntr[1]]. "-1"
			//is for conversion between Fortran one based
			//indices and c++ zero based indices.
			for(int n = 0; n < basisSize; n++)
				b.at(n, 0) = workd[(ipntr[0] - 1) + n];

			luSolver.solve(b);

			for(int n = 0; n < basisSize; n++)
				workd[(ipntr[1] - 1) + n] = b.at(n, 0);

			break;
		default:
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"Unknown mode.",
				"This should never happen, contact the"
				<< " developer."
			);
		}
	}
	else if(ido == 99){
		return true;
	}
	else{
		TBTKExit(
			"ArnoldiIterator::arnoldiLoop()",
			"znaupd returned with ido = '" << ido << "',"
			<< " which is not supported.",
			"This should never happen, contact the"
			<< " developer."
		);
	}

	return false;
}

bool ArnoldiIterator::executeReverseCommunicationMessage(
	int ido,
	int basisSize,
	complex<double> *workd,
	int *ipntr,
	Matrix<complex<double>> &b
){
	if(ido == -1 || ido == 1){
		switch(mode){
		case Mode::Normal:
		{
			//Perform matrix multiplcation y = Ax, where x =
			//workd[ipntr[0]] and y = workd[ipntr[1]]. "-1" is for
			//conversion between Fortran one based indices and c++
			//zero based indices.
			for(int n = 0; n < basisSize; n++)
				workd[(ipntr[1] - 1) + n] = 0.;

/*			const Model &model = getModel();
			int numMatrixElements = model.getHoppingAmplitudeSet().getNumMatrixElements();
			const int *cooRowIndices = model.getHoppingAmplitudeSet().getCOORowIndices();
			const int *cooColIndices = model.getHoppingAmplitudeSet().getCOOColIndices();
			const complex<double> *cooValues = model.getHoppingAmplitudeSet().getCOOValues();
			for(int n = 0; n < numMatrixElements; n++)
				workd[(ipntr[1] - 1) + cooRowIndices[n]] += cooValues[n]*workd[(ipntr[0] - 1) + cooColIndices[n]];*/
			const unsigned int *cscRowIndices = matrix.getCSCRows();
			const unsigned int *cscColPointers = matrix.getCSCColumnPointers();
			const complex<double> *cscValues
				= matrix.getCSCValues();
			for(
				unsigned int column = 0;
				column < matrix.getNumColumns();
				column++
			){
				for(
					unsigned int n = cscColPointers[column];
					n < cscColPointers[column+1];
					n++
				){
					workd[
						(ipntr[1] - 1)
						+ cscRowIndices[n]
					] += cscValues[n]*workd[
						(ipntr[0] - 1)
						+ column
					];
				}
			}

			//Apply shift.
/*			for(int n = 0; n < basisSize; n++)
				workd[(ipntr[1] - 1) + n] -= shift*workd[(ipntr[0] - 1) + n];*/

			break;
		}
		case Mode::ShiftAndInvert:
			//Solve x = (A - sigma*I)^{-1}b, where b =
			//workd[ipntr[0]] and x = workd[ipntr[1]]. "-1"
			//is for conversion between Fortran one based
			//indices and c++ zero based indices.
			for(int n = 0; n < basisSize; n++)
				b.at(n, 0) = workd[(ipntr[0] - 1) + n];

			luSolver.solve(b);

			for(int n = 0; n < basisSize; n++)
				workd[(ipntr[1] - 1) + n] = b.at(n, 0);

			break;
		default:
			TBTKExit(
				"ArnoldiIterator::arnoldiLoop()",
				"Unknown mode.",
				"This should never happen, contact the"
				<< " developer."
			);
		}
	}
	else if(ido == 99){
		return true;
	}
	else{
		TBTKExit(
			"ArnoldiIterator::arnoldiLoop()",
			"znaupd returned with ido = '" << ido << "',"
			<< " which is not supported.",
			"This should never happen, contact the"
			<< " developer."
		);
	}

	return false;
}

void ArnoldiIterator::checkZneupdIerr(int ierr) const{
	if(ierr < 0){
		TBTKExit(
			"ArnoldiIterator::arnoldiLoop()",
			"Input parameter '" << -ierr << "' to zneupd is"
			<< " invalid.",
			"This should never happen, contact the developer."
		);
	}
	else if(ierr != 0){
		TBTKExit(
			"ArnoldiIterator::arnoldiLoop()",
			"zneupd() exited with error ierr = " << ierr
			<< ". Unknown error ().",
			"This should never happen, contact the developer."
		);
		//The documentation for zneupd() says the following in case
		//ierr = 1:
		//
		//The Schur form computed by LAPACK routine csheqr could not be
		//reordered by LAPACK routine ztrsen. Re-enter subroutine
		//ZNEUPD with IPARAM(5)=NCV and increase the size of the array
		//D to have dimension at least dimension NCV and allocate at
		//least NCV columns for Z. NOTE: Not necessary if Z and V share
		//the same space. Please notify the authors if this error
		//occurs.
	}
}

void ArnoldiIterator::initNormal(){
	//Get matrix representation on COO format
/*	const Model &model = getModel();
	const int *cooRowIndices = model.getHoppingAmplitudeSet().getCOORowIndices();
	const int *cooColIndices = model.getHoppingAmplitudeSet().getCOOColIndices();
	TBTKAssert(
		cooRowIndices != NULL && cooColIndices != NULL,
		"ArnoldiIterator::initNormal()",
		"COO format not constructed.",
		"Use Model::constructCOO() to construct COO format."
	);*/

	const Model &model = getModel();

	matrix = SparseMatrix<complex<double>>(
		SparseMatrix<complex<double>>::StorageFormat::CSC
	);

	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= getModel().getHoppingAmplitudeSet().cbegin();
		iterator != getModel().getHoppingAmplitudeSet().cend();
		++iterator
	){
		int from = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getFromIndex()
		);
		int to = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getToIndex()
		);
		matrix.add(to, from, (*iterator).getAmplitude());
	}
	for(int n = 0; n < model.getBasisSize(); n++)
		matrix.add(n, n, -shift);
	matrix.constructCSX();
}

void ArnoldiIterator::initShiftAndInvert(){
	const Model &model = getModel();

	SparseMatrix<complex<double>> matrix(
		SparseMatrix<complex<double>>::StorageFormat::CSC
	);

	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= getModel().getHoppingAmplitudeSet().cbegin();
		iterator != getModel().getHoppingAmplitudeSet().cend();
		++iterator
	){
		int from = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getFromIndex()
		);
		int to = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getToIndex()
		);
		matrix.add(to, from, (*iterator).getAmplitude());
	}
	for(int n = 0; n < model.getBasisSize(); n++)
		matrix.add(n, n, -shift);
	matrix.constructCSX();

	luSolver.setMatrix(matrix);
}

void ArnoldiIterator::sort(){
	CArray<complex<double>> workspace(numEigenValues);
	for(int n = 0; n < numEigenValues; n++)
		workspace[n] = eigenValues[n];

	CArray<int> order(numEigenValues);
	CArray<int> orderWorkspace(numEigenValues);
	for(int n = 0; n < numEigenValues; n++){
		order[n] = n;
		orderWorkspace[n] = n;
	}

	mergeSortSplit(
		workspace.getData(),
		eigenValues.getData(),
		orderWorkspace.getData(),
		order.getData(),
		0,
		numEigenValues
	);

	const Model &model = getModel();
	if(calculateEigenVectors){
		CArray<complex<double>> eigenVectorsWorkspace(numEigenValues*model.getBasisSize());
		for(int n = 0; n < numEigenValues*model.getBasisSize(); n++)
			eigenVectorsWorkspace[n] = eigenVectors[n];

		for(int n = 0; n < numEigenValues; n++)
			for(int c = 0; c < model.getBasisSize(); c++)
				eigenVectors[n*model.getBasisSize() + c] = eigenVectorsWorkspace[order[n]*model.getBasisSize() + c];
	}
}

void ArnoldiIterator::mergeSortSplit(
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

void ArnoldiIterator::mergeSortMerge(
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

};	//End of namespace Solver
};	//End of namespace TBTK
