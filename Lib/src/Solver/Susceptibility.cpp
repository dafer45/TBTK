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
#include "TBTK/Solver/Susceptibility.h"
#include "TBTK/UnitHandler.h"

#include <complex>
#include <iomanip>

using namespace std;

namespace TBTK{
namespace Solver{

Susceptibility::Susceptibility(
	Algorithm algorithm,
	const MomentumSpaceContext &momentumSpaceContext
){
	this->algorithm = algorithm;
	this->momentumSpaceContext = &momentumSpaceContext;

	energiesAreInversionSymmetric = false;

	kPlusQLookupTable = nullptr;
	generateKPlusQLookupTable();

	isMaster = true;
}

Susceptibility::Susceptibility(
	Algorithm algorithm,
	const MomentumSpaceContext &momentumSpaceContext,
	int *kPlusQLookupTable
){
	this->algorithm = algorithm;
	this->momentumSpaceContext = &momentumSpaceContext;

	energiesAreInversionSymmetric = false;

	this->kPlusQLookupTable = kPlusQLookupTable;

	isMaster = false;
}

Susceptibility::~Susceptibility(){
	if(isMaster && kPlusQLookupTable != nullptr)
		delete [] kPlusQLookupTable;
}

void Susceptibility::generateKPlusQLookupTable(){
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

}	//End namespace Solver
}	//End of namesapce TBTK
