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
	eigenVectorsMap = NULL;
	occupancy = NULL;
	numberOfParticles = -1;
	particleNumberIsFixed = true;
	decayMode = DecayMode::None;
	callback = NULL;
	numTimeSteps = 0;
	dt = 0.01;
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
	int basisSize = model->getBasisSize();
	occupancy = new double[basisSize];

	currentTimeStep = -1;
	dSolver.setModel(model);
	dSolver.setSCCallback(scCallback);
	dSolver.run();

	if(numberOfParticles < 0){
		for(int n = 0; n < basisSize; n++){
			numberOfParticles++;
			if(eigenValues[n] >= model->getChemicalPotential())
				break;
		}
	}

	complex<double> *dPsi = new complex<double>[basisSize*basisSize];
	for(int t = 0; t < numTimeSteps; t++){
		currentTimeStep = t;
		callback(this);

		#pragma omp parallel for
		for(int n = 0; n < basisSize*basisSize; n++)
			dPsi[n] = 0.;

		AmplitudeSet::iterator it = model->amplitudeSet.getIterator();
		HoppingAmplitude *ha;
		while((ha = it.getHA())){
			int fromIndex = model->amplitudeSet.getBasisIndex(ha->fromIndex);
			int toIndex = model->amplitudeSet.getBasisIndex(ha->toIndex);
			complex<double> amplitude = ha->getAmplitude();
			#pragma omp parallel for
			for(int n = 0; n < basisSize; n++){
				dPsi[basisSize*n + toIndex] += amplitude*eigenVectorsMap[n][fromIndex];
			}
			it.searchNextHA();
		}

		#pragma omp parallel for
		for(int n = 0; n < basisSize; n++){
			double energy = 0.;
			for(int c = 0; c < basisSize; c++){
				energy += real(conj(eigenVectorsMap[n][c])*dPsi[basisSize*n + c]);
			}
			eigenValues[n] = energy;
		}

		#pragma omp parallel for
		for(int n = 0; n < basisSize; n++){
			for(int c = 0; c < basisSize; c++)
				eigenVectorsMap[n][c] -= i*dPsi[basisSize*n + c]*UnitHandler::convertTimeNtB(dt)/UnitHandler::getHbar();
		}

		sort();

		updateOccupancy();

		#pragma omp parallel for
		for(int n = 0; n < basisSize; n++){
			//No need to use eigenVectorsMap here because
			//noramlization procedure is independent of ordering.
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
			te->onDiagonalizationFinished();
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

void TimeEvolver::onDiagonalizationFinished(){
	eigenValues = dSolver.getEigenValuesRW();
	eigenVectors = dSolver.getEigenVectorsRW();

	if(eigenVectorsMap == NULL){
		int basisSize = model->getBasisSize();
		eigenVectorsMap = new complex<double>*[basisSize];
		for(int n = 0; n < basisSize; n++)
			eigenVectorsMap[n] = &(eigenVectors[n*basisSize]);
	}

	for(int n = 0; n < model->getBasisSize(); n++){
		if(numberOfParticles < 0){
			if(eigenValues[n] < model->getChemicalPotential())
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
}

void TimeEvolver::sort(){
	int basisSize = model->getBasisSize();

	for(int m = 0; m < basisSize; m++){
		for(int n = m+1; n < basisSize; n++){
			if(eigenValues[n] < eigenValues[m]){
				double tempEigenValue = eigenValues[n];
				complex<double> *tempEigenVectorsMap = eigenVectorsMap[n];
				double tempOccupancy = occupancy[n];

				eigenValues[n] = eigenValues[m];
				eigenVectorsMap[n] = eigenVectorsMap[m];
				occupancy[n] = occupancy[m];

				eigenValues[m] = tempEigenValue;
				eigenVectorsMap[m] = tempEigenVectorsMap;
				occupancy[m] = tempOccupancy;
			}
		}
	}
}

void TimeEvolver::updateOccupancy(){
	switch(decayMode){
		case DecayMode::None:
			break;
		case DecayMode::Instantly:
			decayInstantly();
			break;
		case DecayMode::Interpolate:
			decayInterpolate();
			break;
		default:	//Should never happen. Hard error generated for quick bug detection.
			cout << "Error in TimeEvolver::updateOccupancy(): Unkown DecayMode - " << static_cast<int>(decayMode);
			exit(1);
	}
}

void TimeEvolver::decayInstantly(){
	int basisSize = model->getBasisSize();
	if(particleNumberIsFixed){
		for(int n = 0; n < basisSize; n++){
			if(n < numberOfParticles)
				occupancy[n] = 1.;
			else
				occupancy[n] = 0.;
		}
	}
	else{
		numberOfParticles = 0;
		for(int n = 0; n < basisSize; n++){
			if(eigenValues[n] < model->getChemicalPotential()){
				occupancy[n] = 1.;
				numberOfParticles++;
			}
			else{
				occupancy[n] = 0.;
			}
		}
	}
}

void TimeEvolver::decayInterpolate(){
	int basisSize = model->getBasisSize();
	if(particleNumberIsFixed){
		cout << "Error in TimeEvolver::decayInterpolate(): Fixed particle number not supported.\n";
		exit(1);
	}
	else{
		for(int n = 0; n < basisSize; n++){
			if(eigenValues[n] < model->getChemicalPotential()){
				occupancy[n] += 0.00000001;
				if(occupancy[n] > 1)
					occupancy[n] = 1.;
			}
			else{
				occupancy[n] -= 0.00000001;
				if(occupancy[n] < 0)
					occupancy[n] = 0.;
			}
		}
	}
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
