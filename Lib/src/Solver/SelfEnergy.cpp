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

/** @file SelfEnergy.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/InteractionAmplitude.h"
#include "TBTK/Solver/SelfEnergy.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{
namespace Solver{

SelfEnergy::SelfEnergy(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	const Property::InteractionVertex &interactionVertex
) :
	momentumSpaceContext(momentumSpaceContext),
	interactionVertex(interactionVertex)
{
	isInitialized = false;

	kMinusQLookupTable = nullptr;
}

SelfEnergy::~SelfEnergy(){
	if(kMinusQLookupTable != nullptr)
		delete [] kMinusQLookupTable;
}

void SelfEnergy::init(){
	TBTKAssert(
		interactionVertex.getEnergyType()
		== Property::EnergyResolvedProperty<
			complex<double>
		>::EnergyType::BosonicMatsubara,
		"Solver::SelfEnergy::init()",
		"The 'interactionVertex' must have energy type"
		<< "Property::EnergyResolved::Property::EnergyType::BosonicMatsubara.",
		""
	);
	TBTKAssert(
		interactionVertex.getNumMatsubaraEnergies()%2 == 1,
		"Solver::SelfEnergy::init()",
		"The number of summation energies must be an odd number but it"
		<< " is '" << interactionVertex.getNumMatsubaraEnergies()
		<< "'.",
		""
	);
	TBTKAssert(
		interactionVertex.getLowerMatsubaraEnergyIndex()
		== -interactionVertex.getUpperMatsubaraEnergyIndex(),
		"Solver::SelfEnergy::init()",
		"The lower and upper Matsubara energies for the"
		<< " 'interactionVertex' must be each others negatives. The"
		<< " lower Matsubara energy index is '"
		<< interactionVertex.getLowerMatsubaraEnergyIndex() << "',"
		<< " while the upper Matsubara energy index is '"
		<< interactionVertex.getUpperMatsubaraEnergyIndex() << "'.",
		""
	);

	if(kMinusQLookupTable != nullptr){
		delete [] kMinusQLookupTable;
		kMinusQLookupTable = nullptr;
	}

	isInitialized = true;
}

void SelfEnergy::generateKMinusQLookupTable(){
	if(kMinusQLookupTable != nullptr)
		return;

	Timer::tick("Calculate k-q lookup table.");
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const Model &model = getModel();
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
				"Solver::SelfEnergy::generateKMinusQLookupTable()",
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
			"Solver::SelfEnergy::generateKMinusQLookupTable()",
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
inline int SelfEnergy::getKMinusQLinearIndex<false>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	Index kMinusQIndex = momentumSpaceContext.getBrillouinZone().getMinorCellIndex(
		{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
		momentumSpaceContext.getNumMeshPoints()
	);
	return getModel().getHoppingAmplitudeSet().getFirstIndexInBlock(
		kMinusQIndex
	);
}

template<>
inline int SelfEnergy::getKMinusQLinearIndex<true>(
	unsigned int meshIndex,
	const vector<double> &k,
	int kLinearIndex
) const{
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	return kMinusQLookupTable[
		meshIndex*mesh.size() + kLinearIndex/numOrbitals
	];
}

vector<complex<double>> SelfEnergy::calculateSelfEnergy(
	const Index &index,
	const vector<complex<double>> &selfEnergyEnergies
){
	TBTKAssert(
		isInitialized,
		"Solver::SelfEnergy::calculateSelfEnergy()",
		"Solver::SelfEnergy not yet initialized.",
		"Use SelfEnergy::init() to initialize the Solver::SelfEnergy."
	);

	//Initialize results
	vector<complex<double>> result;
	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.push_back(0);

	if(selfEnergyEnergies.size() == 1)
		selfEnergyMainLoop<true>(index, selfEnergyEnergies, result);
	else
		selfEnergyMainLoop<false>(index, selfEnergyEnergies, result);

	return result;
}

vector<complex<double>> SelfEnergy::calculateSelfEnergySelfConsistently(
	unsigned int numMatsubaraFrequencies,
	const vector<complex<double>> &energies
){
	TBTKNotYetImplemented("SelfEnergyCalculator::calculateSelfEnergySelfConsistently");

	TBTKAssert(
		isInitialized,
		"SelfEnergyCalculator::calculateSelfEnergySelfConsistently()",
		"SelfEnergyCalculator not yet initialized.",
		"Use SelfEnergyCalculator::init() to initialize the"
		<< " SelfEnergyCalculator."
	);

	vector<complex<double>> result;

	return result;
}

template<bool singleSelfEnergyEnergy>
void SelfEnergy::selfEnergyMainLoop(
	const Index &index,
	const vector<complex<double>> &selfEnergyEnergies,
	vector<complex<double>> &result
){
	vector<Index> components = index.split();
	TBTKAssert(
		components.size() == 3,
		"Solver::SelfEnergy::calculateSelfEnergy()",
		"The Index must be a compound Index with 3 component Indices,"
		<< " but '" << components.size() << "' components supplied.",
		""
	);

	const Index kIndex = components[0];
	const Index intraBlockIndices0 = components[1];
	const Index intraBlockIndices1 = components[2];

	for(unsigned int n = 0; n < 2; n++){
		TBTKAssert(
			components[n+1].getSize() == 1,
			"Solver::SelfEnergy::calculateSelfEnergy()",
			"The four last components of the compound Index"
			<< " currently is restricted to have a single"
			<< " subindex, but component '" << n+1 << "' has '"
			<< components[n+1].getSize() << "' subindices.",
			"Contact the developer if support for more general"
			<< " Indices is required."
		);
	}

	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	vector<unsigned int> kVector;
	kVector.reserve(kIndex.getSize());
	for(unsigned int n = 0; n < kIndex.getSize(); n++)
		kVector.push_back(kIndex[n]);
	const vector<double> k = brillouinZone.getMinorMeshPoint(
		kVector,
		numMeshPoints
	);

	generateKMinusQLookupTable();

	const RPA::MomentumSpaceContext &momentumSpaceContext
		= this->momentumSpaceContext;
	const Model &model = getModel();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Get linear index corresponding to kIndex.
	int kLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
		kIndex
	);

	const unsigned int NUM_WORKERS = 40;
	vector<vector<complex<double>>> results;
	results.reserve(NUM_WORKERS);
	for(
		unsigned int n = 0;
		n < NUM_WORKERS;
		n++
	){
		results.push_back(vector<complex<double>>());
		results[n].reserve(results.size());
		for(unsigned int c = 0; c < result.size(); c++)
			results[n].push_back(0);
	}

	//Main loop
	#pragma omp parallel for default(none) shared( \
		mesh, \
		kLinearIndex, \
		numOrbitals, \
		momentumSpaceContext, \
		model, \
		results, \
		brillouinZone, \
		numMeshPoints, \
		selfEnergyEnergies \
	) firstprivate( \
		k, \
		intraBlockIndices0, \
		intraBlockIndices1 \
	)
	for(unsigned int worker = 0; worker < NUM_WORKERS; worker++){
		unsigned int blockSize = mesh.size()/NUM_WORKERS;
		unsigned int begin = worker*blockSize;
		unsigned int end = (worker+1)*blockSize;
		if(worker == NUM_WORKERS-1)
			end = mesh.size();

		for(unsigned int n = begin; n < end; n++){
			//Get linear index corresponding to k-q
			int kMinusQLinearIndex = getKMinusQLinearIndex<true>(
				n,
				k,
				kLinearIndex
			);
			int kMinusQMeshPoint = kMinusQLinearIndex/numOrbitals;
			Index qIndex = brillouinZone.getMinorCellIndex(
				mesh[n],
				numMeshPoints
			);

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
					const vector<complex<double>> &selfEnergyVertexData
						= interactionVertex.getData();
					unsigned int offsetSelfEnergyVertex
						= interactionVertex.getOffset({
							qIndex,
							intraBlockIndices0,
							{(int)propagatorEnd},
//							{(int)propagatorEnd},
//							intraBlockIndices0,
							{(int)propagatorStart},
							intraBlockIndices1
						});
					vector<complex<double>> summationEnergies;
					for(
						unsigned int n = 0;
						n < interactionVertex.getNumMatsubaraEnergies();
						n++
					){
						summationEnergies.push_back(
							interactionVertex.getMatsubaraEnergy(n)
						);
					}

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
							e0 < summationEnergies.size();
							e0++
						){
							complex<double> numerator = selfEnergyVertexData[offsetSelfEnergyVertex + e0]*greensFunctionNumerator;
//							complex<double> E = summationEnergies[e0] - relativeStateEnergy;
							complex<double> E = -summationEnergies[e0] - relativeStateEnergy;

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

	for(unsigned int n = 0; n < NUM_WORKERS; n++)
		for(unsigned int c = 0; c < result.size(); c++)
			result[c] += results[n][c];

	//Calculate kT
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			getModel().getTemperature()
		);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;

	for(unsigned int n = 0; n < selfEnergyEnergies.size(); n++)
		result.at(n) *= kT/mesh.size();
}

}	//End of namespace Solver
}	//End of namesapce TBTK
