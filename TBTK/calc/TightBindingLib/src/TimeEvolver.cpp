/** @file TimeEvolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/TimeEvolver.h"
#include "../include/AmplitudeSet.h"
#include "../include/Util.h"
#include <complex>
#include <math.h>

using namespace std;

namespace TBTK{

const complex<double> i(0, 1);

TimeEvolver::TimeEvolver(Model *model){
	this->model = model;
	this->scCallback = NULL;
	numTimeSteps = 0;
	dt = 0.01;
	isAdiabatic = false;
}

TimeEvolver::~TimeEvolver(){
}

void TimeEvolver::run(){
	dSolver.setModel(model);
	dSolver.setSCCallback(scCallback);
	dSolver.run();

	int basisSize = model->getBasisSize();

	double *eigenValues = dSolver.getEigenValuesRW();
	complex<double> *eigenVectors = dSolver.getEigenVectorsRW();
	complex<double> *workspace = new complex<double>[basisSize*basisSize];
	for(int t = 0; t < numTimeSteps; t++){
		tsCallback();

		#pragma omp parallel for
		for(int n = 0; n < basisSize*basisSize; n++)
			workspace[n] = 0.;

		AmplitudeSet::iterator it = model->amplitudeSet.getIterator();
		HoppingAmplitude *ha;
		while((ha = it.getHA())){
			int fromIndex = model->amplitudeSet.getBasisIndex(ha->fromIndex);
			int toIndex = model->amplitudeSet.getBasisIndex(ha->toIndex);
			complex<double> amplitude = ha->getAmplitude();
			#pragma omp parallel for
			for(int n = 0; n < basisSize; n++)
				workspace[basisSize*n + toIndex] += amplitude*eigenVectors[basisSize*n + fromIndex];
			it.searchNextHA();
		}

		if(isAdiabatic){
			#pragma omp parallel for
			for(int n = 0; n < basisSize; n++){
				double energy = 0.;
				for(int c = 0; c < basisSize; c++){
					energy += real(conj(eigenVectors[basisSize*n + c])*workspace[basisSize*n + c]);
				}
				eigenValues[n] = energy;
			}
		}

		#pragma omp parallel for
		for(int n = 0; n < basisSize*basisSize; n++){
			eigenVectors[n] -= i*workspace[n]*dt;
		}

		#pragma omp parallel for
		for(int n = 0; n < basisSize; n++){
			double normalizationFactor = 0.;
			for(int c = 0; c < basisSize; c++){
				normalizationFactor += pow(abs(eigenVectors[n*basisSize + c]), 2);
			}
			normalizationFactor = sqrt(normalizationFactor);
			for(int c = 0; c < basisSize; c++)
				eigenVectors[basisSize*n + c] /= normalizationFactor;
		}

		scCallback(&dSolver);
	}
}

};
