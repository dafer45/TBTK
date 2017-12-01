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

/** @file SusceptibilityCalculator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Functions.h"
#include "RPA/SusceptibilityCalculator.h"
#include "UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

const complex<double> i(0, 1);

namespace TBTK{

SusceptibilityCalculator::SusceptibilityCalculator(
	const MomentumSpaceContext &momentumSpaceContext
){
	this->momentumSpaceContext = &momentumSpaceContext;

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	susceptibilityMode = Mode::Lindhard;
	energyType = EnergyType::Complex;
	susceptibilityEnergiesAreInversionSymmetric = false;
	susceptibilityIsSafeFromPoles = false;

	kPlusQLookupTable = nullptr;
	generateKPlusQLookupTable();

	const Model& model = momentumSpaceContext.getModel();
	fermiDiracLookupTable = new double[model.getBasisSize()];
	for(int n = 0; n < model.getBasisSize(); n++){
		fermiDiracLookupTable[n] = Functions::fermiDiracDistribution(
			momentumSpaceContext.getEnergy(n),
			model.getChemicalPotential(),
			model.getTemperature()
		);
	}

	omp_init_lock(&lindhardLock);

	interactionAmplitudesAreGenerated = false;

	isInitialized = true;
	isMaster = true;
}

SusceptibilityCalculator::SusceptibilityCalculator(
	const MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable,
	double *fermiDiracLookupTable
){
	this->momentumSpaceContext = &momentumSpaceContext;

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	susceptibilityMode = Mode::Lindhard;
	energyType = EnergyType::Complex;
	susceptibilityEnergiesAreInversionSymmetric = false;
	susceptibilityIsSafeFromPoles = false;

	this->kPlusQLookupTable = kPlusQLookupTable;

	this->fermiDiracLookupTable = fermiDiracLookupTable;

	omp_init_lock(&lindhardLock);

	interactionAmplitudesAreGenerated = false;

	isInitialized = true;
	isMaster = false;
}

SusceptibilityCalculator::~SusceptibilityCalculator(){
	if(isMaster && kPlusQLookupTable != nullptr)
		delete [] kPlusQLookupTable;
	if(isMaster && fermiDiracLookupTable != nullptr)
		delete [] fermiDiracLookupTable;
}

SusceptibilityCalculator* SusceptibilityCalculator::createSlave(){
	return new SusceptibilityCalculator(
		*momentumSpaceContext,
		kPlusQLookupTable,
		fermiDiracLookupTable
	);
}

void SusceptibilityCalculator::precompute(unsigned int numWorkers){
	Timer::tick("1");
	const vector<vector<double>> &mesh = momentumSpaceContext->getMesh();
	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	vector<SusceptibilityCalculator*> susceptibilityCalculators;
	for(unsigned int n = 0; n < numWorkers; n++){
		susceptibilityCalculators.push_back(
			new SusceptibilityCalculator(
				*momentumSpaceContext
			)
		);
		susceptibilityCalculators[n]->setSusceptibilityEnergies(
			susceptibilityEnergies
		);
		susceptibilityCalculators[n]->setEnergyType(
			energyType
		);
		susceptibilityCalculators[n]->setSusceptibilityEnergiesAreInversionSymmetric(
			susceptibilityEnergiesAreInversionSymmetric
		);

		delete [] susceptibilityCalculators[n]->kPlusQLookupTable;
		susceptibilityCalculators[n]->kPlusQLookupTable = kPlusQLookupTable;
		delete [] susceptibilityCalculators[n]->fermiDiracLookupTable;
		susceptibilityCalculators[n]->fermiDiracLookupTable = fermiDiracLookupTable;
		susceptibilityCalculators[n]->isMaster = false;
	}

	vector<vector<vector<double>>> meshes;
	unsigned int meshSegmentSize = mesh.size()/(numWorkers-1);
	for(unsigned int n = 0; n < numWorkers-1; n++){
		meshes.push_back(vector<vector<double>>());
		for(
			unsigned int c = n*meshSegmentSize;
			c < (n+1)*meshSegmentSize;
			c++
		){
			meshes[n].push_back(mesh.at(c));
		}
	}
	meshes.push_back(vector<vector<double>>());
	for(
		unsigned int n = (numWorkers-1)*meshSegmentSize;
		n < mesh.size();
		n++
	){
		meshes.back().push_back(mesh.at(n));
	}
	Timer::tock();

	Timer::tick("2");
	#pragma omp parallel for
	for(unsigned int n = 0; n < numWorkers; n++){
		SusceptibilityCalculator &susceptibilityCalculator = *susceptibilityCalculators.at(n);
		const vector<vector<double>> &mesh = meshes.at(n);

		for(unsigned int c = 0; c < mesh.size(); c++){
			for(
				unsigned int orbital0 = 0;
				orbital0 < numOrbitals;
				orbital0++
			){
				for(
					unsigned int orbital1 = 0;
					orbital1 < numOrbitals;
					orbital1++
				){
					for(
						unsigned int orbital2 = 0;
						orbital2 < numOrbitals;
						orbital2++
					){
						for(
							unsigned int orbital3 = 0;
							orbital3 < numOrbitals;
							orbital3++
						){
							susceptibilityCalculator.calculateSusceptibility(
								mesh[c],
								{
									(int)orbital0,
									(int)orbital1,
									(int)orbital2,
									(int)orbital3
								}
							);
						}
					}
				}
			}
		}
	}
	Timer::tock();

	Timer::tick("3");
	Streams::out << "Susceptibility tree size:\t" << susceptibilityTree.getSizeInBytes() << "\n";
	for(unsigned int n = 0; n < numWorkers; n++){
		SusceptibilityCalculator &susceptibilityCalculator = *susceptibilityCalculators.at(n);
		const vector<vector<double>> &mesh = meshes.at(n);

		for(unsigned int c = 0; c < mesh.size(); c++){
			Index kIndex = momentumSpaceContext->getKIndex(
				mesh[c]
			);

			for(
				unsigned int orbital0 = 0;
				orbital0 < numOrbitals;
				orbital0++
			){
				for(
					unsigned int orbital1 = 0;
					orbital1 < numOrbitals;
					orbital1++
				){
					for(
						unsigned int orbital2 = 0;
						orbital2 < numOrbitals;
						orbital2++
					){
						for(
							unsigned int orbital3 = 0;
							orbital3 < numOrbitals;
							orbital3++
						){
							SerializeableVector<complex<double>> result;
							Index resultIndex = Index(
								kIndex,
								{
									(int)orbital0,
									(int)orbital1,
									(int)orbital2,
									(int)orbital3
								}
							);
							TBTKAssert(
								susceptibilityCalculator.susceptibilityTree.get(
									result,
									resultIndex
								),
								"SusceptibilityCalculator::precompute()",
								"Unable to find requested susceptibility.",
								"This should never happen, contact the developer."
							);
							susceptibilityTree.add(
								result,
								resultIndex
							);
						}
					}
				}
			}
		}
	}

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		delete susceptibilityCalculators[n];

	Streams::out << "Susceptibility tree size:\t" << susceptibilityTree.getSizeInBytes() << "\n";
	Timer::tock();
}

void SusceptibilityCalculator::generateKPlusQLookupTable(){
	if(kPlusQLookupTable != nullptr)
		return;

	Timer::tick("Calculate k+q lookup table.");
	const vector<vector<double>> &mesh = momentumSpaceContext->getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext->getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext->getBrillouinZone();
	const Model &model = momentumSpaceContext->getModel();
	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	kPlusQLookupTable = new int[mesh.size()*mesh.size()];

	string cacheName = "cache/kPlusQLookupTable";
	for(
		unsigned int n = 0;
		n < numMeshPoints.size();
		n++
	){
		cacheName += "_" + to_string(numMeshPoints.at(n));
	}
	ifstream fin(cacheName);
	if(fin){
		unsigned int counter = 0;
		int value;
		while(fin >> value){
			TBTKAssert(
				counter < mesh.size()*mesh.size(),
				"SusceptibilityCalculator::generateKPlusQLookupTable()",
				"Found cache file '" << cacheName << "',"
				<< " but it is too large.",
				"Clear the cache to recalculate"
				<< " kPlusQLookupTable."
			);
			kPlusQLookupTable[counter] = value;
			counter++;
		}
		fin.close();

		TBTKAssert(
			counter == mesh.size()*mesh.size(),
			"SusceptibilityCalculator::generateKPlusQLookupTable()",
			"Found cache file" << cacheName << ","
			<< " but it is too small.",
			"Clear the cache to recalculate kPlusQLookupTable."
		);

		Timer::tock();

		return;
	}

	#pragma omp parallel for
	for(unsigned int k = 0; k < mesh.size(); k++){
		const vector<double>& K = mesh.at(k);
		for(unsigned int q = 0; q < mesh.size(); q++){
			vector<double> Q = mesh.at(q);

			vector<double> kPlusQ;
			for(unsigned int n = 0; n < K.size(); n++)
				kPlusQ.push_back(K.at(n)+Q.at(n));

			Index qIndex = brillouinZone.getMinorCellIndex(
				Q,
				numMeshPoints
			);
			int qLinearIndex = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
				qIndex
			);
			Index kPlusQIndex = brillouinZone.getMinorCellIndex(
				kPlusQ,
				numMeshPoints
			);
			kPlusQLookupTable[
				k*mesh.size() + qLinearIndex/numOrbitals
			] = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
				kPlusQIndex
			);
		}
	}

	ofstream fout(cacheName);
	if(fout)
		for(unsigned int n = 0; n < mesh.size()*mesh.size(); n++)
			fout << kPlusQLookupTable[n] << "\n";

	Timer::tock();
}

inline complex<double> SusceptibilityCalculator::getPoleTimesTwoFermi(
	complex<double> energy,
	double e2,
	double e1,
	double chemicalPotential,
	double temperature,
	int kPlusQLinearIndex,
	unsigned int meshPoint,
	unsigned int state2,
	unsigned int state1,
	unsigned int numOrbitals
) const{
	if(abs(energy + e2 - e1) < 1e-10){
		double e = UnitHandler::convertEnergyNtB(
			(e1 + e2)/2. - chemicalPotential
		);
		double t = UnitHandler::convertTemperatureNtB(temperature);
		double kT = UnitHandler::getK_BB()*t;

		//Ratio between derivatives of numerator and denominator
//		return -1./(kT*pow(exp(e/(2.*kT)) + exp(-e/(2.*kT)), 2));
		//Rewrite
		return -1./(kT*pow(cosh(e/(2.*kT))*2., 2));
		//Final rewrite
//		return -1./(kT*(cosh(e/kT) + 1));
	}
	else{
		return (1./(energy + e2 - e1))*(
			fermiDiracLookupTable[
				kPlusQLinearIndex
				+ state2
			]
			- fermiDiracLookupTable[
				meshPoint*numOrbitals
				+ state1
			]
		);
	}
}

//Implementation based on Eq. (14) in
//S. Graser, T. A. Maier, P. J. Hirschfeld, and D. J. Scalapino,
//New Journal of Physics 11, 025016 (2009)
complex<double> SusceptibilityCalculator::calculateSusceptibilityLindhard(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	const vector<vector<double>> &mesh = momentumSpaceContext->getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext->getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext->getBrillouinZone();
	const Model &model = momentumSpaceContext->getModel();
	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	complex<double> result = 0;
	for(unsigned int n = 0; n < mesh.size(); n++){
		Index kPlusQIndex = brillouinZone.getMinorCellIndex(
			{mesh[n][0] + k[0], mesh[n][1] + k[1]},
			numMeshPoints
		);
		int kPlusQLinearIndex =  model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
			kPlusQIndex
		);
		for(unsigned int c = 0; c < numOrbitals; c++){
			double e1 = momentumSpaceContext->getEnergy(n, c);
			complex<double> a1 = momentumSpaceContext->getAmplitude(
				n,
				c,
				orbitalIndices.at(3)
			);
			complex<double> a2 = momentumSpaceContext->getAmplitude(
				n,
				c,
				orbitalIndices.at(0)
			);

			for(unsigned int j = 0; j < numOrbitals; j++){
				double e2 = momentumSpaceContext->getEnergy(
					kPlusQLinearIndex + j
				);

				complex<double> pttf = getPoleTimesTwoFermi(
					energy,
					e2,
					e1,
					model.getChemicalPotential(),
					model.getTemperature(),
					kPlusQLinearIndex,
					n,
					j,
					c,
					numOrbitals
				);

				complex<double> a3 = momentumSpaceContext->getAmplitude(
					kPlusQLinearIndex/numOrbitals,
					j,
					orbitalIndices.at(1)
				);
				complex<double> a4 = momentumSpaceContext->getAmplitude(
					kPlusQLinearIndex/numOrbitals,
					j,
					orbitalIndices.at(2)
				);

				result -= a1*conj(a2)*a3*conj(a4)*pttf;
			}
		}
	}

	result /= mesh.size();

	return result;
}

template<>
inline int SusceptibilityCalculator::getKPlusQLinearIndex<false>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const vector<vector<double>> &mesh = momentumSpaceContext->getMesh();

	Index kPlusQIndex = momentumSpaceContext->getBrillouinZone().getMinorCellIndex(
		{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
		momentumSpaceContext->getNumMeshPoints()
	);
	return momentumSpaceContext->getModel().getHoppingAmplitudeSet()->getFirstIndexInBlock(
		kPlusQIndex
	);
}

template<>
inline int SusceptibilityCalculator::getKPlusQLinearIndex<true>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	return kPlusQLookupTable[
		meshIndex*momentumSpaceContext->getMesh().size()
		+ kLinearIndex/momentumSpaceContext->getNumOrbitals()
	];
}

void SusceptibilityCalculator::cacheSusceptibility(
	const vector<complex<double>> &result,
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	const Index &kIndex,
	const Index &resultIndex
){
	//Cashe result
	susceptibilityTree.add(
		result,
		resultIndex
	);

	const vector<unsigned int> &numMeshPoints = momentumSpaceContext->getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext->getBrillouinZone();

	//<Needs proper checking>
	//Use symmetries to extend result to other entries.
	if(
		getEnergyType() == EnergyType::Imaginary
		&& getSusceptibilityEnergiesAreInversionSymmetric()
	){
		vector<complex<double>> reversedResult;
		vector<complex<double>> conjugatedResult;
		vector<complex<double>> reversedConjugatedResult;
		for(unsigned int n = 0; n < result.size(); n++){
			reversedResult.push_back(
				conj(result.at(result.size()-1-n))
			);
			conjugatedResult.push_back(conj(result.at(n)));
			reversedConjugatedResult.push_back(
				conj(result.at(result.size()-1-n))
			);
		}

		vector<double> kMinus;
		for(unsigned int n = 0; n < k.size(); n++)
			kMinus.push_back(-k.at(n));
		Index kMinusIndex = brillouinZone.getMinorCellIndex(
			kMinus,
			numMeshPoints
		);

		susceptibilityTree.add(
			reversedConjugatedResult,
			Index(
				kIndex,
				{
					orbitalIndices.at(3),
					orbitalIndices.at(2),
					orbitalIndices.at(1),
					orbitalIndices.at(0)
				}
			)
		);
		susceptibilityTree.add(
			reversedResult,
			Index(
				kMinusIndex,
				{
					orbitalIndices.at(2),
					orbitalIndices.at(3),
					orbitalIndices.at(0),
					orbitalIndices.at(1)
				}
			)
		);
		susceptibilityTree.add(
			conjugatedResult,
			Index(
				kMinusIndex,
				{
					orbitalIndices.at(1),
					orbitalIndices.at(0),
					orbitalIndices.at(3),
					orbitalIndices.at(2)
				}
			)
		);
	}
	//</Needs proper checking>
}

template<bool useKPlusQLookupTable, bool isSafeFromPoles>
vector<complex<double>> SusceptibilityCalculator::calculateSusceptibilityLindhard(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	//Get kIndex and resultIndex
	const vector<double> &k = kDual.getContinuousIndex();
	const Index &kIndex = kDual;
	Index resultIndex = getSusceptibilityResultIndex(
		kIndex,
		orbitalIndices
	);

	//Try to return cashed result
	SerializeableVector<complex<double>> result;
	if(susceptibilityTree.get(result, resultIndex))
		return result;

	const vector<vector<double>> &mesh = momentumSpaceContext->getMesh();
	const Model &model = momentumSpaceContext->getModel();
	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	//Initialize result and isolate purely real energies. The real energies
	//need extra careful treatment because they can result in evaluation of
	// a term at a pole.
	vector<complex<double>> realEnergies;
	vector<complex<double>> complexEnergies;
	vector<unsigned int> realEnergyIndices;
	vector<unsigned int> complexEnergyIndices;
	for(unsigned int n = 0; n < susceptibilityEnergies.size(); n++){
		result.push_back(0);
		complex<double> energy = susceptibilityEnergies.at(n);
		if(abs(imag(energy)) < 1e-10){
			realEnergies.push_back(energy);
			realEnergyIndices.push_back(n);
		}
		else{
			complexEnergies.push_back(energy);
			complexEnergyIndices.push_back(n);
		}
	}

	//Get linear index corresponding to kIndex.
	int kLinearIndex = model.getHoppingAmplitudeSet()->getFirstIndexInBlock(
		kIndex
	);

	//Main loop
	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		//Get linear index corresponding to k+q
		int kPlusQLinearIndex = getKPlusQLinearIndex<useKPlusQLookupTable>(
			meshPoint,
			k,
			kLinearIndex
		);
		int kPlusQMeshPoint = kPlusQLinearIndex/momentumSpaceContext->getNumOrbitals();

		for(unsigned int state1 = 0; state1 < numOrbitals; state1++){
			//Get energy and amplitude for first state.
			double e1 = momentumSpaceContext->getEnergy(meshPoint, state1);
			complex<double> a1 = momentumSpaceContext->getAmplitude(
				meshPoint,
				state1,
				orbitalIndices[3]
			);
			complex<double> a2 = momentumSpaceContext->getAmplitude(
				meshPoint,
				state1,
				orbitalIndices[0]
			);

			//Skip to the next state if the current state gives an
			//obvious zero contribution.
/*			if(abs(a1*a2) < 1e-10)
				continue;*/
			if(real(a1*a2*conj(a1*a2)) < 1e-10)
				continue;

			for(
				unsigned int state2 = 0;
				state2 < numOrbitals;
				state2++
			){
				//Get energy and amplitudes for second state
				double e2 = momentumSpaceContext->getEnergy(
					kPlusQLinearIndex + state2
				);
				complex<double> a3 = momentumSpaceContext->getAmplitude(
					kPlusQMeshPoint,
					state2,
					orbitalIndices[1]
				);
				complex<double> a4 = momentumSpaceContext->getAmplitude(
					kPlusQMeshPoint,
					state2,
					orbitalIndices[2]
				);

				//Skip to the next state if the current state
				//gives an obvious zero contribution.
/*				if(abs(a3*a4) < 1e-10)
					continue;*/
				if(real(a3*a4*conj(a3*a4)) < 1e-10)
					continue;

				if(realEnergies.size() > 0){
					//If the expression is not safe from
					//poles, the function
					//poleTimesTwoFermi() is used to
					//evaluate the Lindhard function to
					//properly handle potential divisions
					//by zero.
					for(
						unsigned int e = 0;
						e < realEnergies.size();
						e++
					){
						complex<double> E = realEnergies.at(e);
						complex<double> pttf = getPoleTimesTwoFermi(
							E,
							e2,
							e1,
							model.getChemicalPotential(),
							model.getTemperature(),
							kPlusQLinearIndex,
							meshPoint,
							state2,
							state1,
							numOrbitals
						);

						result[
							realEnergyIndices[e]
						] -= a1*conj(a2)*a3*conj(a4)*pttf;
					}
				}
				if(complexEnergies.size() > 0){
					//If the expression is safe from
					//division by zero, the performance is
					//improved by using a lookup table to
					//evaluate the Fermi function.

					//Calculate numerator of the Lindhard
					//function
					complex<double> numerator = a1*conj(a2)*a3*conj(a4)*(
						fermiDiracLookupTable[
							kPlusQLinearIndex
							+ state2
						]
						- fermiDiracLookupTable[
							meshPoint*numOrbitals
							+ state1
						]
					);
					//Skip to the next state if the
					//numerator gives rise to an obvious
					//zero contribution.
					if(abs(numerator) < 1e-10)
						continue;

					for(
						unsigned int e = 0;
						e < complexEnergies.size();
						e++
					){
						//
						complex<double> E = complexEnergies[e];
						result[
							complexEnergyIndices[e]
						] -= numerator/(E + e2 - e1);
					}
				}
			}
		}
	}

	//Normalize result.
	for(unsigned int n = 0; n < susceptibilityEnergies.size(); n++)
		result[n] /= mesh.size();

	//Cashe result
	cacheSusceptibility(
		result,
		k,
		orbitalIndices,
		kIndex,
		resultIndex
	);

	return result;
}

complex<double> SusceptibilityCalculator::calculateSusceptibility(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy,
	Mode mode
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	switch(mode){
	case Mode::Lindhard:
		return calculateSusceptibilityLindhard(
			k,
			orbitalIndices,
			energy
		);
	case Mode::Matsubara:
		return calculateSusceptibilityMatsubara(
			k,
			orbitalIndices,
			energy
		);
	default:
		TBTKExit(
			"SusceptibilityCalculator::calculateSusceptibility()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
}

vector<complex<double>> SusceptibilityCalculator::calculateSusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	switch(getSusceptibilityMode()){
	case Mode::Lindhard:
		if(kPlusQLookupTable != nullptr){
			if(getSusceptibilityIsSafeFromPoles()){
				return calculateSusceptibilityLindhard<
					true,
					true
				>(
					kDual,
					orbitalIndices
				);
			}
			else{
				return calculateSusceptibilityLindhard<
					true,
					false
				>(
					kDual,
					orbitalIndices
				);
			}
		}
		else{
			if(getSusceptibilityIsSafeFromPoles()){
				return calculateSusceptibilityLindhard<
					false,
					true
				>(
					kDual,
					orbitalIndices
				);
			}
			else{
				return calculateSusceptibilityLindhard<
					false,
					false
				>(
					kDual,
					orbitalIndices
				);
			}
		}
	case Mode::Matsubara:
		return calculateSusceptibilityMatsubara(
			kDual,
			orbitalIndices
		);
	default:
		TBTKExit(
			"SusceptibilityCalculator::calculateSusceptibility()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
}

complex<double> SusceptibilityCalculator::calculateSusceptibilityMatsubara(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	TBTKNotYetImplemented(
		"SusceptibilityCalculator::calculateSusceptibilityMatsubara()"
	);
}

vector<complex<double>> SusceptibilityCalculator::calculateSusceptibilityMatsubara(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKNotYetImplemented(
		"SusceptibilityCalculator::calculateSusceptibilityMatsubara()"
	);
}

extern "C" {
	void zgetrf_(
		int* M,
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		int *info
	);
	void zgetri_(
		int *N,
		complex<double> *A,
		int *lda,
		int *ipiv,
		complex<double> *work,
		int *lwork,
		int *info
	);
}

inline void SusceptibilityCalculator::invertMatrix(
	complex<double> *matrix,
	unsigned int dimensions
){
	int numRows = dimensions;
	int numCols = dimensions;

	int *ipiv = new int[min(numRows, numCols)];
	int lwork = numCols*numCols;
	complex<double> *work = new complex<double>[lwork];
	int info;

	zgetrf_(&numRows, &numCols, matrix, &numRows, ipiv, &info);
	zgetri_(&numRows, matrix, &numRows, ipiv, work, &lwork, &info);

	delete [] ipiv;
	delete [] work;
}

void SusceptibilityCalculator::multiplyMatrices(
	complex<double> *matrix1,
	complex<double> *matrix2,
	complex<double> *result,
	unsigned int dimensions
){
	for(unsigned int n = 0; n < dimensions*dimensions; n++)
		result[n] = 0.;

	for(unsigned int row = 0; row < dimensions; row++)
		for(unsigned int col = 0; col < dimensions; col++)
			for(unsigned int n = 0; n < dimensions; n++)
				result[dimensions*col + row] += matrix1[dimensions*n + row]*matrix2[dimensions*col + n];
}

void printMatrix(complex<double> *matrix, unsigned int dimension){
	for(unsigned int r = 0; r < dimension; r++){
		for(unsigned int c = 0; c < dimension; c++){
			Streams::out << setw(20) << matrix[dimension*c + r];
		}
		Streams::out << "\n";
	}
	Streams::out << "\n";
}

vector<vector<vector<complex<double>>>> SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices,
	const vector<InteractionAmplitude> &interactionAmplitudes
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"SusceptibilityCalculator::rpaSusceptibilityMainAlgorithm()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();
	unsigned int matrixDimension = numOrbitals*numOrbitals;
	//Denominator in the expression chi_RPA = chi_0/(1 - U\chi_0)
	vector<complex<double>*> denominators;

	//Initialize denominator matrices to unit matrices
	for(
		unsigned int e = 0;
		e < susceptibilityEnergies.size();
		e++
	){
		//Create denominator matrix
		denominators.push_back(
			new complex<double>[matrixDimension*matrixDimension]
		);
		//Initialize denominator matrices to unit matrices
		for(
			unsigned int c = 0;
			c < matrixDimension*matrixDimension;
			c++
		){
			denominators.at(e)[c] = 0.;
		}
		for(unsigned int c = 0; c < matrixDimension; c++)
			denominators.at(e)[c*matrixDimension + c] = 1.;
	}

	//Calculate denominator = (1 + U\chi_0)
	for(unsigned int n = 0; n < interactionAmplitudes.size(); n++){
		const InteractionAmplitude &interactionAmplitude = interactionAmplitudes.at(n);

		int c0 = interactionAmplitude.getCreationOperatorIndex(0).at(0);
		int c1 = interactionAmplitude.getCreationOperatorIndex(1).at(0);
		int a0 = interactionAmplitude.getAnnihilationOperatorIndex(0).at(0);
		int a1 = interactionAmplitude.getAnnihilationOperatorIndex(1).at(0);
		complex<double> amplitude = interactionAmplitude.getAmplitude();

		if(abs(amplitude) < 1e-10)
			continue;

		int row = numOrbitals*c0 + a1;
		for(unsigned int c = 0; c < numOrbitals; c++){
			for(unsigned int d = 0; d < numOrbitals; d++){
				int col = numOrbitals*c + d;

				vector<complex<double>> susceptibility = calculateSusceptibility(
					kDual,
					{c1, a0, (int)d, (int)c}
				);
				for(
					unsigned int i = 0;
					i < susceptibilityEnergies.size();
					i++
				){
					denominators.at(i)[
						matrixDimension*col + row
					] += amplitude*susceptibility.at(i);
				}
			}
		}
	}

	//Calculate (1 + U\chi_0)^{-1}
	for(unsigned int n = 0; n < susceptibilityEnergies.size(); n++)
		invertMatrix(denominators.at(n), matrixDimension);

	//Initialize \chi_RPA
	vector<vector<vector<complex<double>>>> rpaSusceptibility;
	for(unsigned int orbital2 = 0; orbital2 < numOrbitals; orbital2++){
		rpaSusceptibility.push_back(vector<vector<complex<double>>>());
		for(unsigned int orbital3 = 0; orbital3 < numOrbitals; orbital3++){
			rpaSusceptibility[orbital2].push_back(vector<complex<double>>());
			for(
				unsigned int e = 0;
				e < susceptibilityEnergies.size();
				e++
			){
				rpaSusceptibility[orbital2][orbital3].push_back(0.);
			}
		}
	}

	//Calculate \chi_RPA = \chi_0/(1 + U\chi_0)
	for(unsigned int c = 0; c < numOrbitals; c++){
		for(unsigned int d = 0; d < numOrbitals; d++){
			vector<complex<double>> susceptibility = calculateSusceptibility(
				kDual,
				{
					orbitalIndices.at(0),
					orbitalIndices.at(1),
					(int)d,
					(int)c
				}
			);
			for(unsigned int orbital2 = 0; orbital2 < numOrbitals; orbital2++){
				for(unsigned int orbital3 = 0; orbital3 < numOrbitals; orbital3++){
					for(
						unsigned int i = 0;
						i < susceptibilityEnergies.size();
						i++
					){
						rpaSusceptibility[orbital2][orbital3].at(i) += denominators.at(i)[
							matrixDimension*(
								numOrbitals*orbital2
								+ orbital3
							) + numOrbitals*c + d
						]*susceptibility.at(i);
					}
				}
			}
		}
	}

	//Free memory allocated for denominators
	for(unsigned int n = 0; n < susceptibilityEnergies.size(); n++)
		delete [] denominators.at(n);

	return rpaSusceptibility;
}

vector<complex<double>> SusceptibilityCalculator::calculateRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cashed result
	SerializeableVector<complex<double>> result;
	if(rpaSusceptibilityTree.get(result, resultIndex))
		return result;

	//Calculate RPA-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudes
	);

	//Cache result
	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

void SusceptibilityCalculator::generateInteractionAmplitudes(){
	if(interactionAmplitudesAreGenerated)
		return;
//	Streams::out << "Generating interaction amplitudes.\n";

	interactionAmplitudesCharge.clear();
	interactionAmplitudesSpin.clear();

	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	//Generate charge-interaction amplitudes.
	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudesCharge.push_back(
			InteractionAmplitude(
				U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					2.*Up - J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					-Up + 2.*J,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudesCharge.push_back(
				InteractionAmplitude(
					Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	//Generate spin-interaction amplitudes.
	for(int a = 0; a < (int)numOrbitals; a++){
		interactionAmplitudesSpin.push_back(
			InteractionAmplitude(
				-U,
				{{a},	{a}},
				{{a},	{a}}
			)
		);

		for(int b = 0; b < (int)numOrbitals; b++){
			if(a == b)
				continue;

			interactionAmplitudesSpin.push_back(
				InteractionAmplitude(
					-J,
					{{a},	{b}},
					{{b},	{a}}
				)
			);
			interactionAmplitudesSpin.push_back(
				InteractionAmplitude(
					-Up,
					{{a},	{b}},
					{{a},	{b}}
				)
			);
			interactionAmplitudesSpin.push_back(
				InteractionAmplitude(
					-Jp,
					{{a},	{a}},
					{{b},	{b}}
				)
			);
		}
	}

	interactionAmplitudesAreGenerated = true;
}

vector<complex<double>> SusceptibilityCalculator::calculateChargeRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cashed result
	SerializeableVector<complex<double>> result;
	if(rpaChargeSusceptibilityTree.get(result, resultIndex))
		return result;

	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate the RPA charge-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesCharge
	);

	//Cache result
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaChargeSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);

			//<!!!Needs proper checking!!!>
			const vector<unsigned int> &numMeshPoints = momentumSpaceContext->getNumMeshPoints();
			const BrillouinZone &brillouinZone = momentumSpaceContext->getBrillouinZone();
			const vector<double> &k = kDual.getContinuousIndex();

			const vector<complex<double>> &result = rpaSusceptibility[orbital0][orbital1];

			//Use symmetries to extend result to other entries.
			if(
				getEnergyType() == EnergyType::Imaginary
				&& getSusceptibilityEnergiesAreInversionSymmetric()
			){
				vector<complex<double>> conjugatedResult;
				for(unsigned int n = 0; n < result.size(); n++){
					conjugatedResult.push_back(
						conj(result.at(n))
					);
				}

				vector<double> kMinus;
				for(unsigned int n = 0; n < k.size(); n++)
					kMinus.push_back(-k.at(n));
				Index kMinusIndex = brillouinZone.getMinorCellIndex(
					kMinus,
					numMeshPoints
				);

				rpaChargeSusceptibilityTree.add(
					conjugatedResult,
					Index(
						kMinusIndex,
						{
							orbitalIndices.at(1),
							orbitalIndices.at(0),
							(int)orbital1,
							(int)orbital0
						}
					)
				);
			}
			//</!!!Needs proper checking!!!>
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

vector<complex<double>> SusceptibilityCalculator::calculateSpinRPASusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SusceptibilityCalculator::calculateSusceptibility()",
		"SusceptibilityCalculator not yet initialized.",
		"Use SusceptibilityCalculator::init() to initialize the"
		<< " SusceptibilityCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	//Get kIndex and resultIndex
	Index resultIndex = getSusceptibilityResultIndex(
		kDual,
		orbitalIndices
	);

	//Try to return cached result
	SerializeableVector<complex<double>> result;
	if(rpaSpinSusceptibilityTree.get(result, resultIndex))
		return result;

	unsigned int numOrbitals = momentumSpaceContext->getNumOrbitals();

	//Setup InteractionAmplitude
	generateInteractionAmplitudes();

	//Calculate RPA spin-susceptibility
	vector<vector<vector<complex<double>>>> rpaSusceptibility = rpaSusceptibilityMainAlgorithm(
		kDual,
		orbitalIndices,
		interactionAmplitudesSpin
	);

	//Cache result
	for(unsigned int orbital0 = 0; orbital0 < numOrbitals; orbital0++){
		for(unsigned int orbital1 = 0; orbital1 < numOrbitals; orbital1++){
			Index resultIndex = getSusceptibilityResultIndex(
				kDual,
				{
					orbitalIndices[0],
					orbitalIndices[1],
					(int)orbital0,
					(int)orbital1
				}
			);
			rpaSpinSusceptibilityTree.add(
				rpaSusceptibility[orbital0][orbital1],
				resultIndex
			);

			//<!!!Needs proper checking!!!>
			const vector<unsigned int> &numMeshPoints = momentumSpaceContext->getNumMeshPoints();
			const BrillouinZone &brillouinZone = momentumSpaceContext->getBrillouinZone();
			const vector<double> &k = kDual.getContinuousIndex();

			const vector<complex<double>> &result = rpaSusceptibility[orbital0][orbital1];

			//Use symmetries to extend result to other entries.
			if(
				getEnergyType() == EnergyType::Imaginary
				&& getSusceptibilityEnergiesAreInversionSymmetric()
			){
				vector<complex<double>> conjugatedResult;
				for(unsigned int n = 0; n < result.size(); n++){
					conjugatedResult.push_back(
						conj(result.at(n))
					);
				}

				vector<double> kMinus;
				for(unsigned int n = 0; n < k.size(); n++)
					kMinus.push_back(-k.at(n));
				Index kMinusIndex = brillouinZone.getMinorCellIndex(
					kMinus,
					numMeshPoints
				);

				rpaSpinSusceptibilityTree.add(
					conjugatedResult,
					Index(
						kMinusIndex,
						{
							orbitalIndices.at(1),
							orbitalIndices.at(0),
							(int)orbital1,
							(int)orbital0
						}
					)
				);
			}
			//</!!!Needs proper checking!!!>
		}
	}
	result = rpaSusceptibility[orbitalIndices[2]][orbitalIndices[3]];

	return result;
}

}	//End of namesapce TBTK
