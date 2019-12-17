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

/** @file SelfEnergyCalculator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/InteractionAmplitude.h"
#include "TBTK/RPA/SelfEnergyCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

const complex<double> i(0, 1);

namespace TBTK{

SelfEnergyCalculator::SelfEnergyCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	unsigned int numWorkers
){
	TBTKAssert(
		numWorkers > 0,
		"SelfEnergyCalculator::SelfEnergyCalculator()",
		"'numWorkers' must be larger than zero.",
		""
	);

	isInitialized = false;

	kMinusQLookupTable = nullptr;

	U = 0.;
	Up = 0.;
	J = 0.;
	Jp = 0.;

	electronFluctuationVertexCalculators.push_back(
		new ElectronFluctuationVertexCalculator(momentumSpaceContext)
	);
	for(unsigned int n = 1; n < numWorkers; n++){
		electronFluctuationVertexCalculators.push_back(
			electronFluctuationVertexCalculators[0]->createSlave()
		);
	}
}

SelfEnergyCalculator::~SelfEnergyCalculator(){
	if(kMinusQLookupTable != nullptr)
		delete [] kMinusQLookupTable;
}

void SelfEnergyCalculator::init(){
	TBTKAssert(
		numSummationEnergies%2 == 1,
		"SelfEnergyCalculator::int()",
		"The number of summation energies must be an odd number.",
		"Use SelfEnergyCalculator::setNumSummationEnergies() to set"
		<< " the number of summation energies."
	);

	if(kMinusQLookupTable != nullptr){
		delete [] kMinusQLookupTable;
		kMinusQLookupTable = nullptr;
	}

	//Calculate kT
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			electronFluctuationVertexCalculators[0]->getMomentumSpaceContext(
		).getModel().getTemperature()
	);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;

	//Initialize summation energies
	for(
		int n = -(int)numSummationEnergies/2;
		n <= (int)numSummationEnergies/2;
		n++
	){
		summationEnergies.push_back(i*M_PI*2.*(double)(n)*kT);
	}
	for(unsigned int n = 0; n < electronFluctuationVertexCalculators.size(); n++){
		electronFluctuationVertexCalculators[n]->setEnergies(
			summationEnergies
		);
		electronFluctuationVertexCalculators[n]->setEnergyType(
			ElectronFluctuationVertexCalculator::EnergyType::Imaginary
		);
		electronFluctuationVertexCalculators[n]->setEnergiesAreInversionSymmetric(
			true
		);
	}

	isInitialized = true;
}

void SelfEnergyCalculator::generateKMinusQLookupTable(){
	if(kMinusQLookupTable != nullptr)
		return;

	Timer::tick("Calculate k-q lookup table.");
	const RPA::MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	kMinusQLookupTable = new int[mesh.size()*mesh.size()];

	string cacheName = "cache/kMinusQLookupTable";
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
				"SelfEnergyCalculator::generateKMinusQLookupTable()",
				"Found cache file '" << cacheName << "',"
				<< " but the cache is too large.",
				"Clear the cache to recalculate"
				<< " kMinusQLookupTable."
			);
			kMinusQLookupTable[counter] = value;
			counter++;
		}
		fin.close();

		TBTKAssert(
			counter == mesh.size()*mesh.size(),
			"SelfEnergyCalculator::generateKMinusQLookupTable()",
			"Found cache file '" << cacheName << "',"
			<< " but the cache is too small.",
			"Clear the cache to recalculate kMinusQLookupTable."
		);

		Timer::tock();

		return;
	}

#ifdef TBTK_USE_OPEN_MP
	#pragma omp parallel for
#endif
	for(unsigned int k = 0; k < mesh.size(); k++){
		const vector<double> &K = mesh.at(k);
		for(unsigned int q = 0; q < mesh.size(); q++){
			const vector<double> &Q = mesh.at(q);

			vector<double> kMinusQ;
			for(unsigned int n = 0; n < K.size(); n++)
				kMinusQ.push_back(K.at(n) - Q.at(n));

			Index qIndex = brillouinZone.getMinorCellIndex(
				Q,
				numMeshPoints
			);
			int qLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
				qIndex
			);
			Index kMinusQIndex = brillouinZone.getMinorCellIndex(
				kMinusQ,
				numMeshPoints
			);
			kMinusQLookupTable[
				k*mesh.size() + qLinearIndex/numOrbitals
			] = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
				kMinusQIndex
			);
		}
	}

	ofstream fout(cacheName);
	if(fout)
		for(unsigned int n = 0; n < mesh.size()*mesh.size(); n++)
			fout << kMinusQLookupTable[n] << "\n";

	Timer::tock();
}

template<>
inline int SelfEnergyCalculator::getKMinusQLinearIndex<false>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const RPA::MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	Index kMinusQIndex = momentumSpaceContext.getBrillouinZone().getMinorCellIndex(
		{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
		momentumSpaceContext.getNumMeshPoints()
	);
	return momentumSpaceContext.getModel().getHoppingAmplitudeSet().getFirstIndexInBlock(
		kMinusQIndex
	);
}

template<>
inline int SelfEnergyCalculator::getKMinusQLinearIndex<true>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const RPA::MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	return kMinusQLookupTable[
		meshIndex*mesh.size() + kLinearIndex/numOrbitals
	];
}

vector<complex<double>> SelfEnergyCalculator::calculateSelfEnergy(
	const vector<double> &k,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergy()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);
	TBTKAssert(
		orbitalIndices.size() == 2,
		"calculateSelfEnergy()",
		"Two orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->generateInteractionAmplitudes();
	}

	const RPA::MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();

	//Calculate offset
	Index kIndex = brillouinZone.getMinorCellIndex(
		k,
		numMeshPoints
	);

	Index resultIndex = Index(
		kIndex,
		{
			orbitalIndices.at(0),
			orbitalIndices.at(1)
		}
	);

	SerializableVector<complex<double>> result;
	if(selfEnergyTree.get(result, resultIndex))
		return result;

	//Initialize results
	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.push_back(0);

	if(selfEnergyEnergies.size() == 1)
		selfEnergyMainLoop<true>(k, orbitalIndices, result);
	else
		selfEnergyMainLoop<false>(k, orbitalIndices, result);

	selfEnergyTree.add(
		result,
		resultIndex
	);

	return result;
}

vector<complex<double>> SelfEnergyCalculator::calculateSelfEnergySelfConsistently(
	unsigned int numMatsubaraFrequencies
){
	TBTKNotYetImplemented("SelfEnergyCalculator::calculateSelfEnergySelfConsistently");

	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergySelfConsistently()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->generateInteractionAmplitudes();
	}

//	const MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
//	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
//	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();

	//Calculate offset
/*	Index kIndex = brillouinZone.getMinorCellIndex(
		k,
		numMeshPoints
	);

	Index resultIndex = Index(
		kIndex,
		{
			orbitalIndices.at(0),
			orbitalIndices.at(1)
		}
	);*/

	SerializableVector<complex<double>> result;
/*	if(selfEnergyTree.get(result, resultIndex))
		return result;

	//Initialize results
	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.push_back(0);

	if(selfEnergyEnergies.size() == 1)
		selfEnergyMainLoop<true>(k, orbitalIndices, result);
	else
		selfEnergyMainLoop<false>(k, orbitalIndices, result);

	selfEnergyTree.add(
		result,
		resultIndex
	);*/

	return result;
}

template<bool singleSelfEnergyEnergy>
void SelfEnergyCalculator::selfEnergyMainLoop(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	vector<complex<double>> &result
){
	generateKMinusQLookupTable();

	const RPA::MomentumSpaceContext &momentumSpaceContext = electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
	const Model &model = momentumSpaceContext.getModel();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Get kIndex
	Index kIndex = momentumSpaceContext.getKIndex(k);

	//Get linear index corresponding to kIndex.
	int kLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
		kIndex
	);

	vector<vector<complex<double>>> results;
	results.reserve(electronFluctuationVertexCalculators.size());
	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		results.push_back(vector<complex<double>>());
		results[n].reserve(results.size());
		for(unsigned int c = 0; c < result.size(); c++)
			results[n].push_back(0);
	}

	//Main loop
	#pragma omp parallel for default(none) shared(mesh, kLinearIndex, k, numOrbitals, orbitalIndices, momentumSpaceContext, model, results)
	for(unsigned int worker = 0; worker < electronFluctuationVertexCalculators.size(); worker++){
		unsigned int blockSize = mesh.size()/electronFluctuationVertexCalculators.size();
		unsigned int begin = worker*blockSize;
		unsigned int end = (worker+1)*blockSize;
		if(worker == electronFluctuationVertexCalculators.size()-1)
			end = mesh.size();

		for(unsigned int n = begin; n < end; n++){
			//Get linear index corresponding to k-q
			int kMinusQLinearIndex = getKMinusQLinearIndex<true>(
				n,
				k,
				kLinearIndex
			);
			int kMinusQMeshPoint = kMinusQLinearIndex/numOrbitals;

			for(
				unsigned int propagatorStart = 0;
				propagatorStart < numOrbitals;
				propagatorStart++
			){
				for(
					unsigned int propagatorEnd = 0;
					propagatorEnd < numOrbitals;
					propagatorEnd++
				){
					vector<complex<double>> selfEnergyVertex
						= electronFluctuationVertexCalculators[worker]->calculateSelfEnergyVertex(
							mesh.at(n),
							{
								(int)propagatorEnd,
								orbitalIndices[0],
								(int)propagatorStart,
								orbitalIndices[1]
							}
						);

					for(
						unsigned int state = 0;
						state < numOrbitals;
						state++
					){
						double e = momentumSpaceContext.getEnergy(
							kMinusQLinearIndex + state
						);
						complex<double> a0 = momentumSpaceContext.getAmplitude(
							kMinusQMeshPoint,
							state,
							propagatorEnd
						);
						complex<double> a1 = momentumSpaceContext.getAmplitude(
							kMinusQMeshPoint,
							state,
							propagatorStart
						);

						complex<double> greensFunctionNumerator = a0*conj(a1);
						double relativeStateEnergy = e - model.getChemicalPotential();

						for(
							unsigned int e0 = 0;
							e0 < numSummationEnergies;
							e0++
						){
							complex<double> numerator = selfEnergyVertex[e0]*greensFunctionNumerator;
							complex<double> E = summationEnergies[e0] - relativeStateEnergy;

							if(singleSelfEnergyEnergy){
								results[worker][0] += numerator/(
									selfEnergyEnergies[0] + E
								);
							}
							else{
								for(
									unsigned int e1 = 0;
									e1 < selfEnergyEnergies.size();
									e1++
								){
									results[worker][e1] += numerator/(
										selfEnergyEnergies[e1] + E
									);
								}
							}
						}
					}
				}
			}
		}
	}

	for(unsigned int n = 0; n < electronFluctuationVertexCalculators.size(); n++)
		for(unsigned int c = 0; c < result.size(); c++)
			result[c] += results[n][c];

	//Calculate kT
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			electronFluctuationVertexCalculators[0]->getMomentumSpaceContext(
		).getModel().getTemperature()
	);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;

	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.at(n) *= kT/mesh.size();
}

}	//End of namesapce TBTK
