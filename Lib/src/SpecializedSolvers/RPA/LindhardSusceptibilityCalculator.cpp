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
#include "TBTK/RPA/LindhardSusceptibilityCalculator.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{

LindhardSusceptibilityCalculator::LindhardSusceptibilityCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext
) :
	SusceptibilityCalculator(Algorithm::Lindhard, momentumSpaceContext)
{
	susceptibilityIsSafeFromPoles = false;

	const Model& model = momentumSpaceContext.getModel();
	fermiDiracLookupTable = new double[model.getBasisSize()];
	for(int n = 0; n < model.getBasisSize(); n++){
		fermiDiracLookupTable[n] = Functions::fermiDiracDistribution(
			momentumSpaceContext.getEnergy(n),
			model.getChemicalPotential(),
			model.getTemperature()
		);
	}
}

LindhardSusceptibilityCalculator::LindhardSusceptibilityCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable,
	double *fermiDiracLookupTable
) :
	SusceptibilityCalculator(
		Algorithm::Lindhard,
		momentumSpaceContext,
		kPlusQLookupTable
	)
{
	susceptibilityIsSafeFromPoles = false;

	this->fermiDiracLookupTable = fermiDiracLookupTable;
}

LindhardSusceptibilityCalculator::~LindhardSusceptibilityCalculator(){
	if(getIsMaster() && fermiDiracLookupTable != nullptr)
		delete [] fermiDiracLookupTable;
}

LindhardSusceptibilityCalculator* LindhardSusceptibilityCalculator::createSlave(){
	return new LindhardSusceptibilityCalculator(
		getMomentumSpaceContext(),
		getKPlusQLookupTable(),
		fermiDiracLookupTable
	);
}

void LindhardSusceptibilityCalculator::precompute(unsigned int numWorkers){
	TBTKNotYetImplemented("LindhardSusceptibilityCalculator::precompute()");
/*	Timer::tick("1");
	const MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	vector<SusceptibilityCalculator*> susceptibilityCalculators;
	for(unsigned int n = 0; n < numWorkers; n++){
		susceptibilityCalculators.push_back(
			new LindhardSusceptibilityCalculator(
				momentumSpaceContext
			)
//			this->createSlave()
		);
		susceptibilityCalculators[n]->setEnergies(
			getEnergies()
		);
		susceptibilityCalculators[n]->setEnergyType(
			getEnergyType()
		);
		susceptibilityCalculators[n]->setEnergiesAreInversionSymmetric(
			getEnergiesAreInversionSymmetric()
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
//	Streams::out << "Susceptibility tree size:\t" << susceptibilityTree.getSizeInBytes() << "\n";
	for(unsigned int n = 0; n < numWorkers; n++){
		SusceptibilityCalculator &susceptibilityCalculator = *susceptibilityCalculators.at(n);
		const vector<vector<double>> &mesh = meshes.at(n);

		for(unsigned int c = 0; c < mesh.size(); c++){
			Index kIndex = momentumSpaceContext.getKIndex(
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
							SerializableVector<complex<double>> result;
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
	Timer::tock();*/
}

inline complex<double> LindhardSusceptibilityCalculator::getPoleTimesTwoFermi(
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
		double e = UnitHandler::convertNaturalToBase<Quantity::Energy>(
			(e1 + e2)/2. - chemicalPotential
		);
		double t = UnitHandler::convertNaturalToBase<
			Quantity::Temperature
		>(temperature);
		double kT = UnitHandler::getConstantInBaseUnits("k_B")*t;

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
complex<double> LindhardSusceptibilityCalculator::calculateSusceptibilityLindhard(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	const RPA::MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	complex<double> result = 0;
	for(unsigned int n = 0; n < mesh.size(); n++){
		Index kPlusQIndex = brillouinZone.getMinorCellIndex(
			{mesh[n][0] + k[0], mesh[n][1] + k[1]},
			numMeshPoints
		);
		int kPlusQLinearIndex =  model.getHoppingAmplitudeSet().getFirstIndexInBlock(
			kPlusQIndex
		);
		for(unsigned int c = 0; c < numOrbitals; c++){
			double e1 = momentumSpaceContext.getEnergy(n, c);
			complex<double> a1 = momentumSpaceContext.getAmplitude(
				n,
				c,
				orbitalIndices.at(3)
			);
			complex<double> a2 = momentumSpaceContext.getAmplitude(
				n,
				c,
				orbitalIndices.at(0)
			);

			for(unsigned int j = 0; j < numOrbitals; j++){
				double e2 = momentumSpaceContext.getEnergy(
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

				complex<double> a3 = momentumSpaceContext.getAmplitude(
					kPlusQLinearIndex/numOrbitals,
					j,
					orbitalIndices.at(1)
				);
				complex<double> a4 = momentumSpaceContext.getAmplitude(
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

template<bool useKPlusQLookupTable, bool isSafeFromPoles>
vector<complex<double>> LindhardSusceptibilityCalculator::calculateSusceptibilityLindhard(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	//Get kIndex and resultIndex
//	const vector<double> &k = kDual.getContinuousIndex();
	const vector<double> &k = kDual;
	const Index &kIndex = kDual;
	Index resultIndex = getSusceptibilityResultIndex(
		kIndex,
		orbitalIndices
	);

	//Try to return cashed result
	SerializableVector<complex<double>> result;
	if(getSusceptibilityTree().get(result, resultIndex))
		return result;

	const RPA::MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Initialize result and isolate purely real energies. The real energies
	//need extra careful treatment because they can result in evaluation of
	// a term at a pole.
	vector<complex<double>> realEnergies;
	vector<complex<double>> complexEnergies;
	vector<unsigned int> realEnergyIndices;
	vector<unsigned int> complexEnergyIndices;
	const vector<complex<double>> &energies = getEnergies();
	for(unsigned int n = 0; n < energies.size(); n++){
		result.push_back(0);
		complex<double> energy = energies.at(n);
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
	int kLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
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
		int kPlusQMeshPoint = kPlusQLinearIndex/momentumSpaceContext.getNumOrbitals();

		for(unsigned int state1 = 0; state1 < numOrbitals; state1++){
			//Get energy and amplitude for first state.
			double e1 = momentumSpaceContext.getEnergy(meshPoint, state1);
			complex<double> a1 = momentumSpaceContext.getAmplitude(
				meshPoint,
				state1,
				orbitalIndices[3]
			);
			complex<double> a2 = momentumSpaceContext.getAmplitude(
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
				double e2 = momentumSpaceContext.getEnergy(
					kPlusQLinearIndex + state2
				);
				complex<double> a3 = momentumSpaceContext.getAmplitude(
					kPlusQMeshPoint,
					state2,
					orbitalIndices[1]
				);
				complex<double> a4 = momentumSpaceContext.getAmplitude(
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
	for(unsigned int n = 0; n < energies.size(); n++)
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

complex<double> LindhardSusceptibilityCalculator::calculateSusceptibility(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	return calculateSusceptibilityLindhard(
		k,
		orbitalIndices,
		energy
	);
}

vector<complex<double>> LindhardSusceptibilityCalculator::calculateSusceptibility(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
	TBTKAssert(
		orbitalIndices.size() == 4,
		"getSusceptibility()",
		"Four orbital indices required but " << orbitalIndices.size()
		<< " supplied.",
		""
	);

	if(getKPlusQLookupTable() != nullptr){
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
}

}	//End of namesapce TBTK

