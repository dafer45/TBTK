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

#include "TBTK/Functions.h"
#include "TBTK/RPA/SusceptibilityCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{

SusceptibilityCalculator::SusceptibilityCalculator(
	Algorithm algorithm,
	const RPA::MomentumSpaceContext &momentumSpaceContext
){
	this->algorithm = algorithm;
	this->momentumSpaceContext = &momentumSpaceContext;

	energyType = EnergyType::Complex;
	energiesAreInversionSymmetric = false;

	kPlusQLookupTable = nullptr;
	generateKPlusQLookupTable();

	isMaster = true;
}

SusceptibilityCalculator::SusceptibilityCalculator(
	Algorithm algorithm,
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable
){
	this->algorithm = algorithm;
	this->momentumSpaceContext = &momentumSpaceContext;

	energyType = EnergyType::Complex;
	energiesAreInversionSymmetric = false;

	this->kPlusQLookupTable = kPlusQLookupTable;

	isMaster = false;
}

SusceptibilityCalculator::~SusceptibilityCalculator(){
	if(isMaster && kPlusQLookupTable != nullptr)
		delete [] kPlusQLookupTable;
}

/*void SusceptibilityCalculator::precompute(unsigned int numWorkers){
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
		susceptibilityCalculators[n]->setEnergies(
			energies
		);
		susceptibilityCalculators[n]->setEnergyType(
			energyType
		);
		susceptibilityCalculators[n]->setEnergiesAreInversionSymmetric(
			energiesAreInversionSymmetric
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
}*/

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

#ifdef TBTK_USE_OPEN_MP
	#pragma omp parallel for
#endif
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
			int qLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
				qIndex
			);
			Index kPlusQIndex = brillouinZone.getMinorCellIndex(
				kPlusQ,
				numMeshPoints
			);
			kPlusQLookupTable[
				k*mesh.size() + qLinearIndex/numOrbitals
			] = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
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

/*template<>
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
}*/

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
		&& getEnergiesAreInversionSymmetric()
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

}	//End of namesapce TBTK

