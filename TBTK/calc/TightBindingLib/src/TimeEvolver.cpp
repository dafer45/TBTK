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

vector<TimeEvolver*> TimeEvolver::timeEvolvers;
vector<DiagonalizationSolver*> TimeEvolver::dSolvers;

TimeEvolver::TimeEvolver(Model *model){
	this->model = model;
	eigenValues = NULL;
	eigenVectors = NULL;
	occupancy = NULL;
	numberOfParticles = -1;
	particleNumberIsFixed = true;
	callback = NULL;
	numTimeSteps = 0;
	dt = 0.01;
	isAdiabatic = false;
	currentTimeStep = -1;
	orthogonalityError = 0.;
	orthogonalityCheckInterval = 0;

	dSolvers.push_back(&dSolver);
	timeEvolvers.push_back(this);
}

TimeEvolver::~TimeEvolver(){
	int timeEvolverIndex = -1;
	for(unsigned int n = 0; n < timeEvolvers.size(); n++){
		if(timeEvolvers.at(n) == this){
			timeEvolverIndex = n;
			break;
		}
	}

	if(timeEvolverIndex != -1){
		timeEvolvers.erase(timeEvolvers.begin() + timeEvolverIndex);
		dSolvers.erase(dSolvers.begin() + timeEvolverIndex);
	}
	else{
		cout << "Error in TimeEvolver::~TimeEvolver(): TimeEvolver not found.\n";
		exit(1);
	}
}

void TimeEvolver::run(){
	currentTimeStep = -1;
	dSolver.setModel(model);
	dSolver.setSCCallback(scCallback);
	dSolver.run();

	int basisSize = model->getBasisSize();

	eigenValues = dSolver.getEigenValuesRW();
	eigenVectors = dSolver.getEigenVectorsRW();
	occupancy = new double[basisSize];
	for(int n = 0; n < basisSize; n++){
		if(numberOfParticles < 0){
			if(eigenValues[n] < 0)
				occupancy[n] = 1.;
			else
				occupancy[n] = 0.;
		}
		else{
			if(n < numberOfParticles)
				occupancy[n] = 1.;
			else
				occupancy[n] = 0.;
		}
	}
	complex<double> *workspace = new complex<double>[basisSize*basisSize];
	for(int t = 0; t < numTimeSteps; t++){
		currentTimeStep = t;
		callback(this);

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
			eigenVectors[n] -= i*workspace[n]*UnitHandler::convertTimeNtB(dt)/UnitHandler::getHbar();
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

		if(orthogonalityCheckInterval != 0 && t%orthogonalityCheckInterval == 0)
			calculateOrthogonalityError();
	}
}

bool TimeEvolver::scCallback(DiagonalizationSolver *dSolver){
	for(unsigned int n = 0; n < dSolvers.size(); n++){
		if(dSolvers.at(n) == dSolver){
			TimeEvolver *te = timeEvolvers.at(n);
			if(te->callback != NULL)
				return te->callback(te);
			else
				return true;	//No self-consistency set, so report that self-consistency has been reached.
		}
	}

	cout << "Error in TimeEvolver::scCallback(): DiagonalizationSolver not found.\n";
	exit(1);

	return 0; //Never reached
}

void TimeEvolver::calculateOrthogonalityError(){
	int basisSize = model->getBasisSize();
	complex<double> *eigenVectors = dSolver.getEigenVectorsRW();

	double maxOverlap = 0;
	for(int i = 0; i < basisSize; i++){
		for(int j = 0; j < basisSize; j++){
			if(i == j)
				continue;

			complex<double> overlap;
			for(int k = 0; k < basisSize; k++){
				overlap += conj(eigenVectors[basisSize*i + k])*eigenVectors[basisSize*j + k];
			}

			if(abs(overlap) > maxOverlap)
				maxOverlap = abs(overlap);
		}
	}

	if(maxOverlap > orthogonalityError)
		orthogonalityError = maxOverlap;
}

};
