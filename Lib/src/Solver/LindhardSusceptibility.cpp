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

/** @file LindhardSusceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Functions.h"
#include "TBTK/Solver/LindhardSusceptibility.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

LindhardSusceptibility::LindhardSusceptibility(
	const RPA::MomentumSpaceContext &momentumSpaceContext
) :
	Susceptibility(Algorithm::Lindhard, momentumSpaceContext)
{
	const Model& model = momentumSpaceContext.getModel();
	fermiDiracLookupTable = new double[model.getBasisSize()];
	for(int n = 0; n < model.getBasisSize(); n++){
		fermiDiracLookupTable[n] = Functions::fermiDiracDistribution(
			momentumSpaceContext.getEnergy(n),
			model.getChemicalPotential(),
			model.getTemperature()
		);
	}

	kPlusQLookupTable = nullptr;
	generateKPlusQLookupTable();

	isMaster = true;
}

LindhardSusceptibility::LindhardSusceptibility(
	const RPA::MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable,
	double *fermiDiracLookupTable
) :
	Susceptibility(
		Algorithm::Lindhard,
		momentumSpaceContext/*,
		kPlusQLookupTable*/
	)
{
	this->fermiDiracLookupTable = fermiDiracLookupTable;
	this->kPlusQLookupTable = kPlusQLookupTable;

	isMaster = false;
}

LindhardSusceptibility::~LindhardSusceptibility(){
	if(isMaster && fermiDiracLookupTable != nullptr)
		delete [] fermiDiracLookupTable;
	if(isMaster && kPlusQLookupTable != nullptr)
		delete [] kPlusQLookupTable;
}

LindhardSusceptibility* LindhardSusceptibility::createSlave(){
	return new LindhardSusceptibility(
		getMomentumSpaceContext(),
		getKPlusQLookupTable(),
		fermiDiracLookupTable
	);
}

inline complex<double> LindhardSusceptibility::getPoleTimesTwoFermi(
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
/*complex<double> LindhardSusceptibility::calculateSusceptibilityLindhard(
	const vector<double> &k,
	const vector<int> &orbitalIndices,
	complex<double> energy
){
	const MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
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
}*/

template<bool useKPlusQLookupTable>
vector<complex<double>> LindhardSusceptibility::calculateSusceptibilityLindhard(
	const Index &index,
	const vector<complex<double>> &energies
){
	std::vector<Index> indices = index.split();
	TBTKAssert(
		indices.size() == 5,
		"LindhardSusceptibility::calculateSusceptibilityLindhard()",
		"The Index must be a compound Index with 5 component Indices,"
		<< " but '" << indices.size() << "' components supplied.",
		""
	);
	const Index kIndex = indices[0];
	for(unsigned int n = 1; n < 5; n++){
		//Temporary restriction that should be removed in the future.
		TBTKAssert(
			indices[n].getSize() == 1,
			"LindhardSusceptibility::calculateSusceptibilityLindhard()",
			"Only single subindex orbitals supported so far.",
			""
		);
	}
	int orbitalIndices[4] = {
		indices[1][0],
		indices[2][0],
		indices[3][0],
		indices[4][0]
	};

	vector<complex<double>> result;

	const RPA::MomentumSpaceContext &momentumSpaceContext = getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints = momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone = momentumSpaceContext.getBrillouinZone();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

	//Initialize result and isolate purely real energies. The real energies
	//need extra careful treatment because they can result in evaluation of
	// a term at a pole.
	vector<complex<double>> realEnergies;
	vector<complex<double>> complexEnergies;
	vector<unsigned int> realEnergyIndices;
	vector<unsigned int> complexEnergyIndices;
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

	vector<unsigned int> kVector;
	kVector.reserve(kIndex.getSize());
	for(unsigned int n = 0; n < kIndex.getSize(); n++)
		kVector.push_back(kIndex[n]);
	vector<double> k = brillouinZone.getMinorMeshPoint(
		kVector,
		numMeshPoints
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

	return result;
}

vector<complex<double>> LindhardSusceptibility::calculateSusceptibility(
	const Index &index,
	const std::vector<std::complex<double>> &energies
){
	if(getKPlusQLookupTable() != nullptr){
		return calculateSusceptibilityLindhard<
			true
		>(
			index,
			energies
		);
	}
	else{
		return calculateSusceptibilityLindhard<
			false
		>(
			index,
			energies
		);
	}
}

void LindhardSusceptibility::generateKPlusQLookupTable(){
	if(kPlusQLookupTable != nullptr)
		return;

	Timer::tick("Calculate k+q lookup table.");
	const RPA::MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();
	const vector<vector<double>> &mesh = momentumSpaceContext.getMesh();
	const vector<unsigned int> &numMeshPoints
		= momentumSpaceContext.getNumMeshPoints();
	const BrillouinZone &brillouinZone
		= momentumSpaceContext.getBrillouinZone();
	const Model &model = momentumSpaceContext.getModel();
	unsigned int numOrbitals = momentumSpaceContext.getNumOrbitals();

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
				"LindhardSusceptibility::generateKPlusQLookupTable()",
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
			"LindhardSusceptibility::generateKPlusQLookupTable()",
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

}	//End of namespace Solver
}	//End of namesapce TBTK
