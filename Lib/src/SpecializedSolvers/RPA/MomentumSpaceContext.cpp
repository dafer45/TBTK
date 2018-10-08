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

/** @file MomentumSpaceContext.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/RPA/MomentumSpaceContext.h"

using namespace std;

namespace TBTK{
namespace RPA{

MomentumSpaceContext::MomentumSpaceContext(){
	brillouinZone = nullptr;
	numOrbitals = 0;
	propertyExtractor = nullptr;
	energies = nullptr;
	amplitudes = nullptr;
	isInitialized = false;
}

MomentumSpaceContext::~MomentumSpaceContext(){
	if(propertyExtractor != nullptr)
		delete propertyExtractor;
	if(energies != nullptr)
		delete [] energies;
	if(amplitudes != nullptr)
		delete [] amplitudes;
}

void MomentumSpaceContext::init(){
	TBTKAssert(
		model != nullptr,
		"MomentumSpaceContext::init()",
		"Model not set.",
		"Use MomentumSpaceContext::setModel() to set the Model."
	);

	TBTKAssert(
		brillouinZone != nullptr,
		"MomentumSpaceContext::init()",
		"BrillouinZone not set.",
		"Use MomentumSpaceContext::setBrillouinZone() to set the"
		<< " BrillouinZone."
	);
	TBTKAssert(
		numMeshPoints.size() == brillouinZone->getNumDimensions(),
		"MomentumSpaceContext::init()",
		"The mesh dimensions must agree with the BrillouinZone"
		<< " dimension. 'numMeshPoints' has " << numMeshPoints.size()
		<< " dimensions" << " but the BrillouinZone has "
		<< brillouinZone->getNumDimensions() << " dimensions.",
		"Use MomentumSpaceContext::setBrillouinZone() to set the"
		<< " number of mesh points."
	);
	mesh = brillouinZone->getMinorMesh(numMeshPoints);

	TBTKAssert(
		numOrbitals != 0,
		"MomentumSpaceContext::init()",
		"The number of orbitals must be larger than 0.",
		"Use MomentumSpaceContext::setNumOrbitals() to set the"
		<< " number of orbitals."
	);

	TBTKAssert(
		(int)(mesh.size()*numOrbitals) == model->getBasisSize(),
		"MomentumSpaceContext::init()",
		"Mesh and orbital specification does not match the Model. The"
		<< " Model has basis size " << model->getBasisSize() << " but"
		<< " mesh.size()*numOrbitals is " << mesh.size()*numOrbitals
		<< ".",
		"Ensure the same number of mesh points are passed to"
		<< " MomentumSpaceContext::setNumMeshPoints() as was used"
		<< " to create the BrillouinZone mesh on which the model was"
		<< " defined. Also make sure the correct number of orbitals"
		<< " are set using MomentumSpaceContext::setNumOrbitals()."
	);

	Timer::tick("Diagonalize");
	solver = Solver::BlockDiagonalizer();
	solver.setModel(*model);
	solver.run();
	Timer::tock();

	if(propertyExtractor != nullptr)
		delete propertyExtractor;
	propertyExtractor = new PropertyExtractor::BlockDiagonalizer(solver);

	if(energies != nullptr)
		delete [] energies;
	energies = new double[model->getBasisSize()];

	if(amplitudes != nullptr)
		delete [] amplitudes;
	amplitudes = new complex<double>[model->getBasisSize()*numOrbitals];

	for(unsigned int meshPoint = 0; meshPoint < mesh.size(); meshPoint++){
		vector<double> k = mesh.at(meshPoint);
		Index kIndex = brillouinZone->getMinorCellIndex(
			k,
			numMeshPoints
		);

		for(
			unsigned int orbital = 0;
			orbital < numOrbitals;
			orbital++
		){
			energies[meshPoint*numOrbitals + orbital]
				= propertyExtractor->getEigenValue(
					kIndex,
					orbital
				);
		}

		for(unsigned int state = 0; state < numOrbitals; state++){
			for(
				unsigned int orbital1 = 0;
				orbital1 < numOrbitals;
				orbital1++
			){
				amplitudes[
					meshPoint*numOrbitals*numOrbitals
					+ state*numOrbitals
					+ orbital1
				] = propertyExtractor->getAmplitude(
					kIndex,
					state,
					{(int)orbital1}
				);
			}
		}
	}

	isInitialized = true;
}

}	//End of namespace RPA
}	//End of namesapce TBTK
