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

/** @file DiagonalizationSolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "DiagonalizationSolver.h"
#include "TBTKMacros.h"
#include "Streams.h"
#include "TBTKMacros.h"

using namespace std;

namespace TBTK{

DiagonalizationSolver::DiagonalizationSolver(){
	model = NULL;

	hamiltonian = NULL;
	eigenValues = NULL;
	eigenVectors = NULL;

	maxIterations = 50;
	scCallback = NULL;
}

DiagonalizationSolver::~DiagonalizationSolver(){
	if(hamiltonian != NULL)
		delete [] hamiltonian;
	if(eigenValues != NULL)
		delete [] eigenValues;
	if(eigenVectors != NULL)
		delete eigenVectors;
}

void DiagonalizationSolver::run(){
	TBTKAssert(
		model != NULL,
		"DiagonalizationSolver::run()",
		"Model not set.",
		"Use DiagonalizationSolver::setModel() to set model."
	);

	int iterationCounter = 0;
	init();

	Streams::out << "Running DiagonalizationSolver\n";
	while(iterationCounter++ < maxIterations){
		if(iterationCounter%10 == 1)
			Streams::out << " ";
		if(iterationCounter%50 == 1)
			Streams::out << "\n";
		Streams::out << "." << flush;

		solve();

		if(scCallback){
			if(scCallback(this))
				break;
			else
				update();
		}
		else{
			break;
		}
	}
	Streams::out << "\n";
}

void DiagonalizationSolver::init(){
	Streams::out << "Initializing DiagonalizationSolver\n";

//	model->amplitudeSet.construct();

	int basisSize = model->getBasisSize();
	Streams::out << "\tBasis size: " << basisSize << "\n";

	hamiltonian = new complex<double>[(basisSize*(basisSize+1))/2];
	eigenValues = new double[basisSize];
	eigenVectors = new complex<double>[basisSize*basisSize];

	update();
}

void DiagonalizationSolver::update(){
	int basisSize = model->getBasisSize();

	for(int n = 0; n < (basisSize*(basisSize+1))/2; n++)
		hamiltonian[n] = 0.;

	AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
	const HoppingAmplitude *ha;
	while((ha = it.getHA())){
		int from = model->getAmplitudeSet()->getBasisIndex(ha->fromIndex);
		int to = model->getAmplitudeSet()->getBasisIndex(ha->toIndex);
		if(from >= to)
			hamiltonian[to + (from*(from+1))/2] += ha->getAmplitude();

		it.searchNextHA();
	}
}

//Lapack function for matrix diagonalization of triangular matrix.
extern "C" void zhpev_(char *jobz,		//'E' = Eigenvalues only, 'V' = Eigenvalues and eigenvectors.
			char *uplo,		//'U' = Stored as upper triangular, 'L' = Stored as lower triangular.
			int *n,			//n*n = Matrix size
			complex<double> *ap,	//Input matrix
			double *w,		//Eigenvalues, is in accending order if info = 0
			complex<double> *z,	//Eigenvectors
			int *ldz,		//
			complex<double> *work,	//Workspace, dimension = max(1, 2*N-1)
			double *rwork,		//Workspace, dimension = max(1, 3*N-2)
			int *info);		//0 = successful, <0 = -info value was illegal, >0 = info number of off-diagonal elements failed to converge.

//Lapack function for matrix diagonalization of banded triangular matrix
extern "C" void zhbeb_(
	char *jobz,		//'E' = Eigenvalues only, 'V' = Eigenvalues and eigenvectors.
	char *uplo,		//'U' = Stored as upper triangular, 'L' = Stored as lower triangular.
	int *n,			//n*n = Matrix size
	int *kd,		//Number of (sub/super)diagonal elements
	complex<double> *ab,	//Input matrix
	int *ldab,		//Leading dimension of array ab. ldab >= kd + 1
	double *w,		//Eigenvalues, is in accending order if info = 0
	complex<double> *z,	//Eigenvectors
	int *ldz,		//
	complex<double> *work,	//Workspace, dimension = max(1, 2*N-1)
	double *rwork,		//Workspace, dimension = max(1, 3*N-2)
	int *info);		//0 = successful, <0 = -info value was illegal, >0 = info number of off-diagonal elements failed to converge.

void DiagonalizationSolver::solve(){
	if(true){//Currently no support for banded matrices.
		//Setup zhpev to calculate...
		char jobz = 'V';		//...eigenvalues and eigenvectors...
		char uplo = 'U';		//...for an upper triangular...
		int n = model->getBasisSize();	//...nxn-matrix.
		//Initialize workspaces
		complex<double> *work = new complex<double>[(2*n-1)];
		double *rwork = new double[3*n-2];
		int info;
		//Solve brop
		zhpev_(&jobz, &uplo, &n, hamiltonian, eigenValues, eigenVectors, &n, work, rwork, &info);

		TBTKAssert(
			info == 0,
			"DiagonalizationSolver:solve()",
			"Diagonalization routine zhpev exited with INFO=" + to_string(info) + ".",
			"See LAPACK documentation for zhpev for further information."
		);

		//Delete workspaces
		delete [] work;
		delete [] rwork;
	}
/*	else{
		int kd;
		if(size_z != 1)
			kd = orbitals*size_x*size_y*size_z;
		else if(size_y != 1)
			kd = orbitals*size_x*size_y;
		else
			kd = orbitals*size_x;
		//Setup zhbev to calculate...
		char jobz = 'V';			//...eigenvalues and eigenvectors...
		char uplo = 'U';			//...for and upper triangluar...
		int n = orbitals*size_x*size_y*size_z;	//...nxn-matrix.
		int ldab = kd + 1;
		//Initialize workspaces
		complex<double> *work = new complex<double>[n];
		double *rwork = new double[3*n-2];
		int info;
		//Solve brop
		zhbev_(&jobz, &uplo, &n, &kd, hamiltonian, &ldab, eigenValues, eigenVectors, &n, work, rwork, &info);

		//delete workspaces
		delete [] work;
		delete [] rwork;
	}*/
}

};	//End of namespace TBTK
