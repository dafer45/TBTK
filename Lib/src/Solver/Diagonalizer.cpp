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

/** @file Diagonalizer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Solver{

Diagonalizer::Diagonalizer() : Communicator(false){
	maxIterations = 50;
	selfConsistencyCallback = nullptr;
}

void Diagonalizer::run(){
	int iterationCounter = 0;
	init();

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Running Diagonalizer\n";
	while(iterationCounter++ < maxIterations){
		if(getGlobalVerbose() && getVerbose()){
			if(iterationCounter%10 == 1)
				Streams::out << " ";
			if(iterationCounter%50 == 1)
				Streams::out << "\n";
			Streams::out << "." << flush;
		}

		solve();

		if(selfConsistencyCallback){
			if(selfConsistencyCallback->selfConsistencyCallback(*this))
				break;
			else
				update();
		}
		else{
			break;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";
}

void Diagonalizer::init(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "Initializing Diagonalizer\n";

	int basisSize = getModel().getBasisSize();
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\tBasis size: " << basisSize << "\n";

	hamiltonian = CArray<complex<double>>((basisSize*(basisSize+1))/2);
	eigenValues = CArray<double>(basisSize);
	eigenVectors = CArray<complex<double>>(basisSize*basisSize);

	update();
}

void Diagonalizer::update(){
	const Model &model = getModel();
	int basisSize = model.getBasisSize();

	for(int n = 0; n < (basisSize*(basisSize+1))/2; n++)
		hamiltonian[n] = 0.;

	for(
		HoppingAmplitudeSet::ConstIterator iterator
			= model.getHoppingAmplitudeSet().cbegin();
		iterator != model.getHoppingAmplitudeSet().cend();
		++iterator
	){
		int from = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getFromIndex()
		);
		int to = model.getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getToIndex()
		);
		if(from >= to)
			hamiltonian[to + (from*(from+1))/2] += (*iterator).getAmplitude();
	}

	setupBasisTransformation();
	transformToOrthonormalBasis();
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

void Diagonalizer::setupBasisTransformation(){
	//Get the OverlapAmplitudeSet.
	const OverlapAmplitudeSet &overlapAmplitudeSet
		= getModel().getOverlapAmplitudeSet();

	//Skip if the basis is assumed to be orthonormal.
	if(overlapAmplitudeSet.getAssumeOrthonormalBasis())
		return;

	//Fill the overlap matrix.
	int basisSize = getModel().getBasisSize();
	CArray<complex<double>> overlapMatrix((basisSize*(basisSize+1))/2);
	for(int n = 0; n < (basisSize*(basisSize+1))/2; n++)
		overlapMatrix[n] = 0;

	for(
		OverlapAmplitudeSet::ConstIterator iterator
			= overlapAmplitudeSet.cbegin();
		iterator != overlapAmplitudeSet.cend();
		++iterator
	){
		int row = getModel().getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getBraIndex()
		);
		int col = getModel().getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getKetIndex()
		);
		if(col >= row){
			overlapMatrix[row + (col*(col+1))/2]
				+= (*iterator).getAmplitude();
		}
	}

	//Diagonalize the overlap matrix.
	char jobz = 'V';
	char uplo = 'U';
	int n = basisSize;

	CArray<complex<double>> work(2*n-1);
	CArray<double> rwork(3*n-2);
	int info;

	CArray<double> overlapMatrixEigenValues(basisSize);
	CArray<complex<double>> overlapMatrixEigenVectors(basisSize*basisSize);

	zhpev_(
		&jobz,
		&uplo,
		&n,
		overlapMatrix.getData(),
		overlapMatrixEigenValues.getData(),
		overlapMatrixEigenVectors.getData(),
		&n,
		work.getData(),
		rwork.getData(),
		&info
	);

	//Setup basisTransformation storage.
	basisTransformation = CArray<complex<double>>(basisSize*basisSize);

	//Calculate the basis transformation using canonical orthogonalization.
	//See for example section 3.4.5 in Moder Quantum Chemistry, Attila
	//Szabo and Neil S. Ostlund.
	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			basisTransformation[row + basisSize*col]
				= overlapMatrixEigenVectors[
					row + basisSize*col
				]/sqrt(
					overlapMatrixEigenValues[col]
				);
		}
	}
}

void Diagonalizer::transformToOrthonormalBasis(){
	//Skip if no basis transformation has been set up (the original basis
	//is assumed to be orthonormal).
	if(basisTransformation.getData() == nullptr)
		return;

	int basisSize = getModel().getBasisSize();

	//Perform the transformation H' = U^{\dagger}HU, where U is the
	//transform to the orthonormal basis.
	Matrix<complex<double>> h(basisSize, basisSize);
	Matrix<complex<double>> U(basisSize, basisSize);
	Matrix<complex<double>> Udagger(basisSize, basisSize);
	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			if(col >= row){
				h.at(row, col)
					= hamiltonian[row + (col*(col+1))/2];
			}
			else{
				h.at(row, col) = conj(
					hamiltonian[col + (row*(row+1))/2]
				);
			}

			U.at(row, col)
				= basisTransformation[row + basisSize*col];

			Udagger.at(row, col) = conj(
				basisTransformation[col + basisSize*row]
			);
		}
	}

	Matrix<complex<double>> hp = Udagger*h*U;

	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			if(col >= row){
				hamiltonian[row + (col*(col+1))/2]
					= hp.at(row, col);
			}
		}
	}
}

void Diagonalizer::transformToOriginalBasis(){
	//Skip if no basis transformation has been set up (the original basis
	//is assumed to be orthonormal).
	if(basisTransformation.getData() == nullptr)
		return;

	int basisSize = getModel().getBasisSize();

	//Perform the transformation v = Uv', where U is the transformation to
	//the orthonormal basis and v and v' are the eigenvectors in the
	//original and orthonormal basis, respectively.
	Matrix<complex<double>> U(basisSize, basisSize);
	Matrix<complex<double>> Vp(basisSize, basisSize);
	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			U.at(row, col)
				= basisTransformation[row + basisSize*col];

			Vp.at(row, col) = eigenVectors[row + basisSize*col];
		}
	}

	Matrix<complex<double>> V = U*Vp;

	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			eigenVectors[row + basisSize*col]
				= V.at(row, col);
		}
	}
}

void Diagonalizer::solve(){
	if(true){//Currently no support for banded matrices.
		//Setup zhpev to calculate...
		char jobz = 'V';		//...eigenvalues and eigenvectors...
		char uplo = 'U';		//...for an upper triangular...
		int n = getModel().getBasisSize();	//...nxn-matrix.
		//Initialize workspaces
		CArray<complex<double>> work(2*n-1);
		CArray<double> rwork(3*n-2);
		int info;
		//Solve brop
		zhpev_(
			&jobz,
			&uplo,
			&n,
			hamiltonian.getData(),
			eigenValues.getData(),
			eigenVectors.getData(),
			&n,
			work.getData(),
			rwork.getData(),
			&info
		);

		TBTKAssert(
			info == 0,
			"Diagonalizer:solve()",
			"Diagonalization routine zhpev exited with INFO=" + to_string(info) + ".",
			"See LAPACK documentation for zhpev for further information."
		);
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

	transformToOriginalBasis();
}

};	//End of namespace Solver
};	//End of namespace TBTK
