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

/** @file TimeEvolver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Solver/TimeEvolver.h"

#include <complex>
#include <cmath>

using namespace std;

namespace TBTK{
namespace Solver{

const complex<double> i(0, 1);

vector<TimeEvolver*> TimeEvolver::timeEvolvers;
vector<Diagonalizer*> TimeEvolver::dSolvers;

TimeEvolver::TimeEvolver(){
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
		TBTKExit(
			"TimeEvolver::~TimeEvolver()",
			"TimeEvolver not found.",
			""
		);
	}
}

void TimeEvolver::run(){
	Model &model = getModel();
	int basisSize = model.getBasisSize();
	occupancy = new double[basisSize];

	currentTimeStep = -1;
	dSolver.setModel(model);
	dSolver.setSelfConsistencyCallback(selfConsistencyCallback);
	dSolver.run();

	if(numberOfParticles < 0){
		for(int n = 0; n < basisSize; n++){
			numberOfParticles++;
			if(eigenValues[n] >= model.getChemicalPotential())
				break;
		}
	}

	double hbar = UnitHandler::getConstantInBaseUnits("hbar");
	complex<double> *dPsi = new complex<double>[basisSize*basisSize];
	for(int t = 0; t < numTimeSteps; t++){
		currentTimeStep = t;
		callback(this);

		#pragma omp parallel for
		for(int n = 0; n < basisSize*basisSize; n++)
			dPsi[n] = 0.;

		for(
			HoppingAmplitudeSet::ConstIterator iterator
				= model.getHoppingAmplitudeSet().cbegin();
			iterator != model.getHoppingAmplitudeSet().cend();
			++iterator
		){
			int fromIndex = model.getHoppingAmplitudeSet().getBasisIndex((*iterator).getFromIndex());
			int toIndex = model.getHoppingAmplitudeSet().getBasisIndex((*iterator).getToIndex());
			complex<double> amplitude = (*iterator).getAmplitude();
			#pragma omp parallel for
			for(int n = 0; n < basisSize; n++){
				dPsi[basisSize*n + toIndex] += amplitude*eigenVectorsMap[n][fromIndex];
			}
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
				eigenVectorsMap[n][c] -= i*dPsi[
					basisSize*n + c
				]*UnitHandler::convertNaturalToBase<
					Quantity::Time
				>(dt)/hbar;
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

bool TimeEvolver::SelfConsistencyCallback::selfConsistencyCallback(
	Diagonalizer &dSolver
){
	for(unsigned int n = 0; n < dSolvers.size(); n++){
		if(dSolvers.at(n) == &dSolver){
			TimeEvolver *te = timeEvolvers.at(n);
			te->onDiagonalizationFinished();
			if(te->callback != NULL)
				return te->callback(te);
			else
				return true;	//No self-consistency set, so report that self-consistency has been reached.
		}
	}

	TBTKExit(
		"TimeEvolver::selfConsistencyCallback()",
		"Diagonalizer not found.",
		""
	);

	return 0; //Never reached
}

void TimeEvolver::onDiagonalizationFinished(){
	eigenValues = dSolver.getEigenValuesRW().getData();
	eigenVectors = dSolver.getEigenVectorsRW().getData();

	const Model &model = getModel();
	if(eigenVectorsMap == NULL){
		int basisSize = model.getBasisSize();
		eigenVectorsMap = new complex<double>*[basisSize];
		for(int n = 0; n < basisSize; n++)
			eigenVectorsMap[n] = &(eigenVectors[n*basisSize]);
	}

	for(int n = 0; n < model.getBasisSize(); n++){
		if(numberOfParticles < 0){
			if(eigenValues[n] < model.getChemicalPotential())
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
	int basisSize = getModel().getBasisSize();

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
		case DecayMode::Custom:
			decayHandler->decay(this, occupancy, eigenValues, eigenVectorsMap);
			break;
		default:	//Should never happen. Hard error generated for quick bug detection.
			TBTKExit(
				"TimeEvolver::updateOccupancy()",
				"Unkown DecayMode - " << static_cast<int>(decayMode) << ".",
				""
			);
	}
}

void TimeEvolver::decayInstantly(){
	const Model &model = getModel();
	int basisSize = model.getBasisSize();
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
			if(eigenValues[n] < model.getChemicalPotential()){
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
	TBTKAssert(
		!particleNumberIsFixed,
		"TimeEvolver::decayInterpolate()",
		"Fixed particle number not supported.",
		"Use TimeEvolver::fixParticleNumber(false) to set non fixed particle number."
	);

	const Model &model = getModel();
	int basisSize = model.getBasisSize();

	for(int n = 0; n < basisSize; n++){
		if(eigenValues[n] < model.getChemicalPotential()){
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

void TimeEvolver::calculateOrthogonalityError(){
	int basisSize = getModel().getBasisSize();
	CArray<complex<double>> &eigenVectors = dSolver.getEigenVectorsRW();

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

};	//End of namespace Solver
};	//End of namespace TBTK
