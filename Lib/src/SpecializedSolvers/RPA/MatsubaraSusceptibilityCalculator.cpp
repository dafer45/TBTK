/* Copyright 2018 Kristofer Björnson
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

#include "TBTK/RPA/MatsubaraSusceptibilityCalculator.h"

#include <complex>
#include <iomanip>

using namespace std;

//const complex<double> i(0, 1);

namespace TBTK{

MatsubaraSusceptibilityCalculator::MatsubaraSusceptibilityCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext
) :
	SusceptibilityCalculator(Algorithm::Matsubara, momentumSpaceContext)
{
	greensFunction = nullptr;
}

MatsubaraSusceptibilityCalculator::MatsubaraSusceptibilityCalculator(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable
) :
	SusceptibilityCalculator(
		Algorithm::Matsubara,
		momentumSpaceContext,
		kPlusQLookupTable
	)
{
	greensFunction = nullptr;
}

MatsubaraSusceptibilityCalculator::~MatsubaraSusceptibilityCalculator(){
	if(greensFunction != nullptr)
		delete [] greensFunction;
}

MatsubaraSusceptibilityCalculator* MatsubaraSusceptibilityCalculator::createSlave(){
	return new MatsubaraSusceptibilityCalculator(
		getMomentumSpaceContext(),
		getKPlusQLookupTable()
	);
}

complex<double> MatsubaraSusceptibilityCalculator::calculateSusceptibility(
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

	return calculateSusceptibilityMatsubara(
		k,
		orbitalIndices,
		energy
	);
}

vector<complex<double>> MatsubaraSusceptibilityCalculator::calculateSusceptibility(
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
		return calculateSusceptibilityMatsubara<true>(
			kDual,
			orbitalIndices
		);
	}
	else{
		return calculateSusceptibilityMatsubara<false>(
			kDual,
			orbitalIndices
		);
	}
}

complex<double> MatsubaraSusceptibilityCalculator::calculateSusceptibilityMatsubara(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	TBTKNotYetImplemented(
		"SusceptibilityCalculator::calculateSusceptibilityMatsubara()"
	);
}

template<bool useKPlusQLookupTable>
vector<complex<double>> MatsubaraSusceptibilityCalculator::calculateSusceptibilityMatsubara(
	const DualIndex &kDual,
	const vector<int> &orbitalIndices
){
/*	TBTKAssert(
		getEnergies().size() == summationEnergies.size(),
		"MatsubaraSusceptibilityCalculator::calculateSusceptibilityMatsubara()",
		"Only equally sized 'energies' and 'summationEnergies' are"
		<< " supported yet.",
		""
	);*/

	//Get kIndex and resultIndex
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

	const vector<complex<double>> &energies = getEnergies();
	for(unsigned int n = 0; n < energies.size(); n++)
		result.push_back(0);

	calculateGreensFunction();

	const RPA::MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();
	const Model &model = momentumSpaceContext.getModel();

	//Get linear index corresponding to kIndex.
	int kLinearIndex = model.getHoppingAmplitudeSet().getFirstIndexInBlock(
		kIndex
	);

	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		//Get linear index corresponding to k+q
		int kPlusQLinearIndex = getKPlusQLinearIndex<useKPlusQLookupTable>(
			meshPoint,
			k,
			kLinearIndex
		);

		int kPlusQMeshPoint = kPlusQLinearIndex/momentumSpaceContext.getNumOrbitals();

		for(unsigned int e = 0; e < energies.size(); e++){
			for(unsigned int n = 0; n < summationEnergies.size(); n++){
				if(
					(int)n + (int)e
					- (int)energies.size()/2 < 0
				){
					continue;
				}
				if(
					(int)n + (int)e
					- (int)energies.size()/2
					>= (int)summationEnergies.size()
				){
					continue;
				}

				result[e] -= getGreensFunctionValue(
					meshPoint,
					orbitalIndices[3],
					orbitalIndices[0],
					n,
					summationEnergies.size(),
					numOrbitals
				)*getGreensFunctionValue(
					kPlusQMeshPoint,
					orbitalIndices[1],
					orbitalIndices[2],
					n + e - energies.size()/2,
					summationEnergies.size(),
					numOrbitals
				);
			}
		}
	}

	//Normalize result.
	double temperature
		= UnitHandler::convertNaturalToBase<Quantity::Temperature>(
			momentumSpaceContext.getModel().getTemperature()
		);
	double kT = UnitHandler::getConstantInBaseUnits("k_B")*temperature;
	for(unsigned int n = 0; n < energies.size(); n++)
		result[n] /= mesh.size()*kT;

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

void MatsubaraSusceptibilityCalculator::calculateGreensFunction(){
	if(greensFunction != nullptr)
		return;

	TBTKAssert(
		summationEnergies.size() != 0,
		"MatsubaraSusceptibilityCalculator::calculateGreensFunction()",
		"Number of summation energies cannot be zero.",
		"Use MatsubaraSusceptibilityCalculator::setNumSummationEnergies()"
		<< " to set the number of summation energies."
	);

	const RPA::MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	greensFunction = new complex<double>[
		mesh.size()*numOrbitals*numOrbitals*summationEnergies.size()
	];

	for(
		unsigned int n = 0;
		n < mesh.size()*numOrbitals*numOrbitals*summationEnergies.size();
		n++
	){
		greensFunction[n] = 0.;
	}

	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		for(unsigned int state = 0; state < numOrbitals; state++){
			double energy = momentumSpaceContext.getEnergy(
				meshPoint,
				state
			);
			for(
				unsigned int orbital0 = 0;
				orbital0 < numOrbitals;
				orbital0++
			){
				complex<double> a0 = momentumSpaceContext.getAmplitude(
					meshPoint,
					state,
					orbital0
				);
				for(
					unsigned int orbital1 = 0;
					orbital1 < numOrbitals;
					orbital1++
				){
					complex<double> a1 = momentumSpaceContext.getAmplitude(
						meshPoint,
						state,
						orbital1
					);
					for(
						unsigned int e = 0;
						e < summationEnergies.size();
						e++
					){
						getGreensFunctionValue(
							meshPoint,
							orbital0,
							orbital1,
							e,
							summationEnergies.size(),
							numOrbitals
						) += a0*conj(a1)/(summationEnergies[e] - energy);
					}
				}
			}
		}
	}
}

}	//End of namesapce TBTK
