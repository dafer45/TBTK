/** @file Model.cpp
 *
 *  @author Kristofer Björnson
 */

#include <iostream>
#include "../include/Model.h"
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

Model::Model(int mode, int numEigenstates){
	this->mode = mode;
	this->numEigenstates = numEigenstates;
/*	//Init hamiltonian, eigen_values, and eigen_vectors;
	hamiltonian = NULL;
	eigen_values = NULL;
	eigen_vectors = NULL;

	maxIterations = 50;
	scCallback = NULL;*/
}

Model::~Model(){
/*	if(hamiltonian != NULL)
		delete hamiltonian;
	if(eigen_values != NULL)
		delete eigen_values;
	if(eigen_vectors != NULL)
		delete eigen_vectors;*/
}

void Model::addHA(HoppingAmplitude ha){
	amplitudeSet.addHA(ha);
}

void Model::addHAAndHC(HoppingAmplitude ha){
	amplitudeSet.addHAAndHC(ha);
}

void Model::construct(){
	cout << "Constructing system\n";

	amplitudeSet.construct();

	int basisSize = getBasisSize();
	cout << "\tBasis size: " << basisSize << "\n";

/*	if(mode == MODE_ALL_EIGENVECTORS){
		hamiltonian = new complex<double>[(basisSize*(basisSize+1))/2];
		eigen_values = new double[basisSize];
		eigen_vectors = new complex<double>[basisSize*basisSize];

		update();
	}*/
}

/*void Model::update(){
	int basisSize = getBasisSize();

	for(int n = 0; n < (basisSize*(basisSize+1))/2; n++)
		hamiltonian[n] = 0.;

	AmplitudeSet::iterator it = amplitudeSet.getIterator();
	HoppingAmplitude *ha;
	while((ha = it.getHA())){
		int from = amplitudeSet.getBasisIndex(ha->fromIndex);
		int to = amplitudeSet.getBasisIndex(ha->toIndex);
		if(from >= to){
			hamiltonian[to + (from*(from+1))/2] += ha->getAmplitude();
		}
		it.searchNextHA();
	}
}

void Model::run(){
	int iterationCounter = 0;
	construct();

	cout << "Running";
	while(iterationCounter++ < maxIterations){
		if(iterationCounter%10 == 1)
			cout << " ";
		if(iterationCounter%50 == 1)
			cout << "\n";
		cout << "." << flush;

		solve();

		if(scCallback){
			if(scCallback(this))
				break;
			else
				update();
		}
		else
			break;
	}
	cout << "\n";
}

void Model::setSCCallback(bool (*scCallback)(Model *model)){
	this->scCallback = scCallback;
}

void Model::print(){
	amplitudeSet.print();

	if(hamiltonian != NULL){
		int basisSize = getBasisSize();
		for(int r = 0; r < basisSize; r++){
			for(int c = 0; c < basisSize; c++){
				if(c >= r)
					cout << hamiltonian[r + (c*(c+1))/2] << "\t";
				else
					cout << conj(hamiltonian[c + (r*(r+1))/2]) << "\t";
			}
			cout << endl;
		}
	}
}

//Lapack function for matrix diagonalizeation of triangular matrix.
extern "C" void zhpev_(char* jobz,		//'E' = Eigenvalues only, 'V' = Eigenvalues and eigenvectors.
						char* uplo,		//'U' = Stored as upper triangular, 'L' = Stored as lower triangular.
						int* n,			//n*n = Matrix size
						complex<double>* ap,		//Input matrix
						double* w,		//Eigenvalues, is in acceding order if info = 0
						complex<double>* z,		//Eigenvectors
						int* ldz,		//
						complex<double>* work,	//Workspace, dimension = max(1, 2*N-1)
						double* rwork,	//Workspace,dimension = max(1, 3*N-2)
						int* info);		//0 = successful, <0 = -info value was illegal, >0 = info number of offdiagonal elements failed to converge.

//Lapack function for matrix diagonalizeation of banded triangular matrix.
extern "C" void zhbev_(char* jobz,		//'E' = Eigenvalues only, 'V' = Eigenvalues and eigenvectors.
						char* uplo,		//'U' = Stored as upper triangular, 'L' = Stored as lower triangular.
						int* n,			//n*n = Matrix size
						int* kd,		//Number of (sub/super)diagonal elements
						complex<double>* ab,		//Input matrix
						int* ldab,		//Leading dimension of array ab. ldab >= kd + 1
						double* w,		//Eigenvalues, is in acceding order if info = 0
						complex<double>* z,		//Eigenvectors
						int* ldz,		//
						complex<double>* work,	//Workspace, dimension = max(1, 2*N-1)
						double* rwork,	//Workspace,dimension = max(1, 3*N-2)
						int* info);		//0 = successful, <0 = -info value was illegal, >0 = info number of offdiagonal elements failed to converge.

void Model::solve(){
	if(true){//Currently no support for banded matrices.
		//Setup zhpev to calculate
		char jobz = 'V';	//...eigenvalues and eigen vectors...
		char uplo = 'U';	//...for an upper triangular...
		int n = getBasisSize();	//...nxn-matrix.
		//Initialize workspaces
		complex<double>* work = new complex<double>[(2*n-1)];
		double* rwork = new double[(3*n-2)];
		int info;
		//Solve brop
		zhpev_(&jobz, &uplo, &n, hamiltonian, eigen_values, eigen_vectors, &n, work, rwork, &info);

		//Delete workspaces.
		delete [] work;
		delete [] rwork;
	}*/
/*	else{
		int kd;
		if(size_z != 1)
			kd = orbitals*size_x*size_y + orbitals-1;
		else if(size_y != 1)
			kd = orbitals*size_x + orbitals-1;
		else
			kd = orbitals + orbitals-1;
		//Setup zhbev to calculate
		char jobz = 'V';	//...eigenvalues and eigen vectors...
		char uplo = 'U';	//...for an upper triangular...
		int n = orbitals*size_x*size_y*size_z;	//...nxn-matrix.
		int ldab = kd + 1;
		//Initialize workspaces
		complex<double>* work = new complex<double>[n];
		double* rwork = new double[(3*n-2)];
		int info;
		//Solve brop
		zhbev_(&jobz, &uplo, &n, &kd, hamiltonian, &ldab, eigen_values, eigen_vectors, &n, work, rwork, &info);

		//Delete workspaces.
		delete [] work;
		delete [] rwork;
	}*/
/*}*/

